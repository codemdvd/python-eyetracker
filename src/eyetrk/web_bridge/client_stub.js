// Minimal stub that a browser page can use to send samples to Python
const ws = new WebSocket("ws://localhost:8000/ws/webgazer");
ws.onopen = () => console.log("WS connected");
function sendSample(s){ ws.send(JSON.stringify(s)); }
// call sendSample({tracker_id:"webgazer", session_id, timestamp_ms, x_norm, y_norm, confidence})