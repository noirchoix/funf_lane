$headers = @{ "X-API-Key" = "8b7d3b2e6b1a4d2f9a3c1c7f0a2e9c44" }

$event1 = @{
  project_id = "phytoquery"
  user_id    = "u_123"
  session_id = "s_456"
  run_id     = "run_001"
  event_type = "decision"
  importance = 4
  title      = "Substitution accepted"
  decision   = "Replace clove oil with cardamom oil"
  reason     = "Lower oxidation risk; keeps warm spicy profile"
  tags       = @("substitution","spice","oxidation")
} | ConvertTo-Json -Depth 10 -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/memory/event" `
  -Method Post `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $event1 | ConvertTo-Json -Depth 10