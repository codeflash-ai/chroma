import collections

import grpc
from opentelemetry.trace import StatusCode, SpanKind


class _ClientCallDetails(
    collections.namedtuple(
        "_ClientCallDetails", ("method", "timeout", "metadata", "credentials")
    ),
    grpc.ClientCallDetails,
):
    pass


def _encode_span_id(span_id: int) -> str:
    return span_id.to_bytes(8, "big").hex()


def _encode_trace_id(trace_id: int) -> str:
    return trace_id.to_bytes(16, "big").hex()


# Using OtelInterceptor with gRPC:
# 1. Instantiate the interceptor: interceptors = [OtelInterceptor()]
# 2. Intercept the channel: channel = grpc.intercept_channel(channel, *interceptors)


class OtelInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        from chromadb.telemetry.opentelemetry import tracer

        if tracer is None:
            return continuation(client_call_details, request_or_iterator)
        with tracer.start_as_current_span(
            f"RPC {client_call_details.method}", kind=SpanKind.CLIENT
        ) as span:
            # Prepare metadata for propagation
            orig_metadata = client_call_details.metadata
            if orig_metadata:
                # Avoid slicing by just converting to list if not already
                if isinstance(orig_metadata, tuple):
                    metadata = list(orig_metadata)
                else:
                    metadata = list(orig_metadata)
            else:
                metadata = []
            span_ctx = span.get_span_context()
            metadata.append(
                (
                    "chroma-traceid",
                    _encode_trace_id(span_ctx.trace_id),
                )
            )
            metadata.append(
                (
                    "chroma-spanid",
                    _encode_span_id(span_ctx.span_id),
                )
            )
            # Update client call details with new metadata
            new_client_details = _ClientCallDetails(
                client_call_details.method,
                client_call_details.timeout,
                tuple(metadata),  # Ensure metadata is a tuple
                client_call_details.credentials,
            )
            try:
                result = continuation(new_client_details, request_or_iterator)
                # Set attributes based on the result
                # Reduce hasattr checks by reusing result.code()
                code = result.code()
                if hasattr(result, "details"):
                    details = result.details()
                    if details:
                        span.set_attribute("rpc.detail", details)
                span.set_attribute("rpc.status_code", code.name.lower())
                span.set_attribute("rpc.status_code_value", code.value[0])
                # Set span status based on gRPC call result
                if code != grpc.StatusCode.OK:
                    span.set_status(StatusCode.ERROR, description=str(code))
                return result
            except Exception as e:
                # Log exception details and re-raise
                span.set_attribute("rpc.error", str(e))
                span.set_status(StatusCode.ERROR, description=str(e))
                raise

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(
        self, continuation, client_call_details, request_iterator
    ):
        return self._intercept_call(continuation, client_call_details, request_iterator)

    def intercept_stream_stream(
        self, continuation, client_call_details, request_iterator
    ):
        return self._intercept_call(continuation, client_call_details, request_iterator)
