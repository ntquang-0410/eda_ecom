# Notify — watcher mốc + Telegram

Mục đích: agent chỉ được đánh thức khi một lệnh nền **kết thúc**. Vì vậy watcher
phải là tiến trình **ngắn hạn, exit ngay khi có mốc**, không phải vòng lặp ngủ dài.

## Thành phần

| file | việc |
|---|---|
| `watch_milestones.py` | stream log, exit khi: model xong / kernel terminal / Traceback / hết `--max-minutes` |
| `telegram.py` | gửi tin qua Bot API |
| `watch_and_notify.sh` | chạy watcher rồi đẩy kết quả vào Telegram |

## Cấu hình token

Không commit token. Đặt ngoài repo:

```bash
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/telegram.json <<'JSON'
{"bot_token": "123456789:AA...", "chat_id": 123456789}
JSON
chmod 600 ~/.config/opencode/telegram.json
```

`chat_id` bỏ trống được — `telegram.py --whoami` sẽ dò qua `getUpdates`
(nhưng phải nhắn 1 tin cho bot trước).

Hoặc dùng env: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Dùng

```bash
python kaggle/notify/telegram.py "hello"        # test
python kaggle/notify/telegram.py --whoami       # dò chat_id
kaggle/notify/watch_and_notify.sh ntquang0410/<kernel> "label" /tmp/ms.json 30
```

Mỗi kernel một `state file` riêng (`MILESTONE_STATE`) để nhiều watcher chạy song song
không ghi đè nhau.
