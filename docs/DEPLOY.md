# Deploy — personal-ask-backend

Runs on the same VPS that serves `arifinoid.web.id`, behind nginx.

## systemd unit

`/etc/systemd/system/ask-api.service`:
```ini
[Unit]
Description=ask.arifinoid.web.id FastAPI service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/arifinoid/personal-ask-backend
EnvironmentFile=/var/www/arifinoid/personal-ask-backend/.env
ExecStart=/usr/bin/uvicorn main:app --host 127.0.0.1 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

Commands:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ask-api
sudo systemctl status ask-api
```

## nginx

Inside the `arifinoid.web.id` server block:
```nginx
location /health {
    proxy_pass http://127.0.0.1:8080;
}
location /v1/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

Then `sudo nginx -t && sudo systemctl reload nginx`.

CORS is locked to `ALLOWED_ORIGINS` from `.env`.
