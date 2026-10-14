"""PushPlus 推送的「受理」判定。

PushPlus 的 /send 是异步接口：HTTP 200 只代表服务端收到了请求，响应体里的 code 才是
受理结果，最终投递还要按流水号去查。2026-10-14 那次手动触发的美股扫描，日志印着
「PushPlus 微信通知发送成功」，微信却一条都没收到 —— 因为旧代码只看 HTTP 状态码，
把 code=900/905/500 这类"没发出去"的响应也当成了成功。

这里锁住：受理失败必须返回 False 并把返回码打进日志，受理成功要留下流水号。
"""

from __future__ import annotations

import pytest

from hk_rebound_screener import notifier

URL = "https://www.pushplus.plus/send?token=TESTTOKEN"


class _FakeResponse:
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.text = body


def _patch_post(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> dict:
    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002 - 对齐 requests 签名
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return response

    monkeypatch.setattr(notifier.requests, "post", fake_post)
    return captured


def test_business_error_code_is_a_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """HTTP 200 + code 900：请求被限流，消息不会发出，必须判失败。"""
    _patch_post(monkeypatch, _FakeResponse(200, '{"code":900,"msg":"请求次数过多,减少请求后2天恢复正常","data":null}'))

    assert notifier.send_webhook_notification("正文", title="标题", webhook_url=URL) is False
    out = capsys.readouterr().out
    assert "code=900" in out and "账号受限" in out


def test_unverified_account_code_is_a_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """905 = 未实名认证，同样不能当成发送成功。"""
    _patch_post(monkeypatch, _FakeResponse(200, '{"code":905,"msg":"账户未进行实名认证","data":null}'))

    assert notifier.send_webhook_notification("正文", webhook_url=URL) is False
    assert "905" in capsys.readouterr().out


def test_accepted_response_logs_short_code(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """受理成功：返回 True，并把流水号写进日志，便于回头核对投递结果。"""
    _patch_post(monkeypatch, _FakeResponse(200, '{"code":200,"msg":"请求成功","data":"3cbc5eab19fe512e"}'))

    assert notifier.send_webhook_notification("正文", webhook_url=URL) is True
    out = capsys.readouterr().out
    assert "3cbc5eab19fe512e" in out and "WARN" not in out


def test_http_error_is_a_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _patch_post(monkeypatch, _FakeResponse(502, "bad gateway"))

    assert notifier.send_webhook_notification("正文", webhook_url=URL) is False
    assert "HTTP 502" in capsys.readouterr().out


def test_non_json_body_is_not_treated_as_success(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """非 JSON 响应无法确认受理，宁可判失败也不能打印成功。"""
    _patch_post(monkeypatch, _FakeResponse(200, "<html>gateway</html>"))

    assert notifier.send_webhook_notification("正文", webhook_url=URL) is False
    assert "不是 JSON" in capsys.readouterr().out


def test_missing_token_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有 NOTIFICATION_WEBHOOK 时直接返回 False，不发请求。"""
    monkeypatch.delenv("NOTIFICATION_WEBHOOK", raising=False)

    assert notifier.send_webhook_notification("正文", webhook_url=None) is False


def test_request_carries_markdown_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """顺带锁住请求本身：token 从 URL 里取，模板是 markdown。"""
    captured = _patch_post(monkeypatch, _FakeResponse(200, '{"code":200,"msg":"请求成功","data":"x"}'))

    notifier.send_webhook_notification("正文", title="标题", webhook_url=URL)

    assert captured["url"] == "http://www.pushplus.plus/send"
    assert captured["json"]["token"] == "TESTTOKEN"
    assert captured["json"]["template"] == "markdown"
    assert captured["json"]["title"] == "标题"
