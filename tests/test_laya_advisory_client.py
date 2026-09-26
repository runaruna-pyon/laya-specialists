import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import URLError

from laya_advisory_client import (
    AdvisoryClient,
    AdvisoryProtocolError,
    AdvisoryServiceUnavailable,
)
from laya_advisory_service import (
    FROZEN_CALIBRATION_SHA256,
    FROZEN_MODEL_SHA256,
    FROZEN_POLICY_SHA256,
)


def valid_response():
    return {
        "request_id": "test-id",
        "advisory_status": "advisory_only",
        "decision": {"final_label": "ready_now", "hierarchical_path": []},
        "stages": [],
        "uncertainty": {"by_stage": []},
        "experimental_policy_result": "CLEAR",
        "autonomy_policy": {"status": "experimental_failed_gate", "authorizes_autonomous_execution": False},
        "provenance": {
            "model_sha256": FROZEN_MODEL_SHA256,
            "calibration_sha256": FROZEN_CALIBRATION_SHA256,
            "policy_sha256": FROZEN_POLICY_SHA256,
            "runtime_version": "test",
        },
    }


class PayloadHandler(BaseHTTPRequestHandler):
    payload = valid_response()
    status = 200

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok", "model_loaded": True,
                           "model_sha256": FROZEN_MODEL_SHA256,
                           "calibration_sha256": FROZEN_CALIBRATION_SHA256,
                           "policy_sha256": FROZEN_POLICY_SHA256}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.dumps(self.payload).encode()
        self.send_response(self.status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AdvisoryClientTests(unittest.TestCase):
    def setUp(self):
        PayloadHandler.payload = valid_response()
        PayloadHandler.status = 200
        self.server = HTTPServer(("127.0.0.1", 0), PayloadHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_client_json_roundtrip_and_autonomy_contract(self):
        client = AdvisoryClient(self.base_url, timeout=2)
        response = client.decide("state text", task="task label", request_id="r-1")
        self.assertEqual(response["request_id"], "test-id")
        self.assertFalse(response["autonomy_policy"]["authorizes_autonomous_execution"])

    def test_health_checks_frozen_provenance(self):
        health = AdvisoryClient(self.base_url, timeout=2).health()
        self.assertTrue(health["model_loaded"])

    def test_client_rejects_non_loopback_base_url(self):
        with self.assertRaises(ValueError):
            AdvisoryClient("http://example.com:8765")

    def test_client_rejects_provenance_mismatch(self):
        PayloadHandler.payload["provenance"]["model_sha256"] = "0" * 64
        client = AdvisoryClient(self.base_url, timeout=2)
        with self.assertRaises(AdvisoryProtocolError):
            client.decide("state")

    def test_health_rejects_provenance_mismatch(self):
        PayloadHandler.payload = valid_response()
        client = AdvisoryClient(self.base_url, timeout=2)
        # Health and decision provenance are checked independently; mutate the test server response.
        original = PayloadHandler.do_GET
        def bad_health(handler):
            body = json.dumps({"status": "ok", "model_loaded": True,
                               "model_sha256": "0" * 64,
                               "calibration_sha256": FROZEN_CALIBRATION_SHA256,
                               "policy_sha256": FROZEN_POLICY_SHA256}).encode()
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        PayloadHandler.do_GET = bad_health
        try:
            with self.assertRaises(AdvisoryProtocolError):
                client.health()
        finally:
            PayloadHandler.do_GET = original

    def test_client_rejects_missing_advisory_authorization_guard(self):
        del PayloadHandler.payload["autonomy_policy"]
        client = AdvisoryClient(self.base_url, timeout=2)
        with self.assertRaises(AdvisoryProtocolError):
            client.decide("state")

    def test_client_surfaces_non_false_authorization_as_protocol_error(self):
        PayloadHandler.payload["autonomy_policy"]["authorizes_autonomous_execution"] = True
        client = AdvisoryClient(self.base_url, timeout=2)
        with self.assertRaises(AdvisoryProtocolError):
            client.decide("state")

    def test_client_reports_unavailable_service(self):
        self.server.shutdown()
        self.server.server_close()
        client = AdvisoryClient("http://127.0.0.1:1", timeout=0.1)
        with self.assertRaises(AdvisoryServiceUnavailable):
            client.decide("state")


if __name__ == "__main__":
    unittest.main()
