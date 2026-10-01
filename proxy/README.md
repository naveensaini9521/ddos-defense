# Proxy Layer

- **nginx**   — best for L7 HTTP parsing, `limit_req`, `limit_conn`
- **haproxy** — best for L4/L7 load balancing, high throughput
- **envoy**   — best for observability + xDS dynamic config

For this lab: start with nginx, add haproxy, then envoy.
