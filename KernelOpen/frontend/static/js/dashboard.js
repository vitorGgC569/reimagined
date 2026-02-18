// UHK Dashboard Visualizer
// Renders "Extremely Detailed Sprites" for Kernel Simulation

const canvas = document.getElementById('gridCanvas');
const ctx = canvas.getContext('2d');

// Resize handling
function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}
window.addEventListener('resize', resize);
resize();

// --- STATE ---
let kernelData = {
    status: "IDLE",
    active_sms: 0,
    throughput: 0,
    ring_buffer_usage: 0
};

// --- SPRITE GENERATION (Procedural Pixel Art) ---
// We draw "chips" representing SMs (Streaming Multiprocessors)
function drawSM(x, y, size, active, intensity) {
    ctx.fillStyle = active ? '#14141e' : '#0a0a0f';
    ctx.fillRect(x, y, size, size); // Base

    // Chip border
    ctx.strokeStyle = active ? (intensity > 0.8 ? '#ff0055' : '#00ff9d') : '#333';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, size, size);

    // Inner Circuitry (Pixel Detail)
    if (active) {
        ctx.fillStyle = intensity > 0.8 ? '#ff0055' : '#00ff9d';
        const pixelSize = size / 8;

        // Random blink pattern based on time
        const t = Date.now() / 100;

        // Draw "Core" pixels
        ctx.fillRect(x + pixelSize*2, y + pixelSize*2, pixelSize*4, pixelSize*4);

        // Draw "Traces"
        if (Math.sin(t + x) > 0) ctx.fillRect(x + pixelSize, y + pixelSize, pixelSize, pixelSize);
        if (Math.cos(t + y) > 0) ctx.fillRect(x + size - pixelSize*2, y + size - pixelSize*2, pixelSize, pixelSize);

        // Glow effect (Canvas Shadow)
        ctx.shadowBlur = 10;
        ctx.shadowColor = ctx.fillStyle;
    } else {
        ctx.shadowBlur = 0;
    }

    ctx.shadowBlur = 0; // Reset
}

// --- MAIN LOOP ---
function draw() {
    ctx.fillStyle = '#050505'; // Deep black background
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Grid of SMs (132 SMs typical for H100)
    // Layout: 12 x 11 grid
    const cols = 16;
    const rows = 9;
    const pad = 10;
    const size = Math.min((canvas.width - pad*cols)/cols, (canvas.height - pad*rows)/rows) - pad;

    let activeCount = kernelData.active_sms;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const x = pad + c * (size + pad);
            const y = pad + r * (size + pad);

            const isActive = idx < activeCount;
            // Simulated "Heat" per core
            const intensity = Math.random();

            drawSM(x, y, size, isActive, intensity);
        }
    }

    // Draw "Ring Buffer" Stream Overlay
    // A flowing stream of data packets at the bottom
    drawStream(canvas.height - 30);

    requestAnimationFrame(draw);
}

function drawStream(y) {
    const usage = kernelData.ring_buffer_usage; // 0-100
    const count = Math.floor((usage / 100) * 20); // Number of packets
    const t = Date.now() / 5;

    ctx.fillStyle = '#00ccff';
    for(let i=0; i<count; i++) {
        let x = (t + i * 100) % canvas.width;
        ctx.fillRect(x, y, 20, 10);
        ctx.shadowColor = '#00ccff';
        ctx.shadowBlur = 10;
        ctx.fillRect(x, y, 20, 10);
        ctx.shadowBlur = 0;
    }
}

// --- API POLLING ---
async function updateStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        kernelData = data;

        // Update DOM Elements
        document.getElementById('status-display').innerText = data.status;
        document.getElementById('status-display').style.color = data.status === "RUNNING" ? "#00ff9d" : "#e0e0e0";

        document.getElementById('tops-display').innerText = data.throughput.toFixed(1) + " TOPS";
        document.getElementById('lat-display').innerText = data.latency.toFixed(2) + " us";
        document.getElementById('rb-display').innerText = data.ring_buffer_usage + "%";
        document.getElementById('sms-display').innerText = data.active_sms + " / 132";

        // Bars
        document.getElementById('sms-bar').style.width = (data.active_sms / 132 * 100) + "%";
        document.getElementById('rb-bar').style.width = data.ring_buffer_usage + "%";

    } catch(e) {
        console.error(e);
    }
    setTimeout(updateStatus, 500);
}

// --- CONTROLS ---
window.startJob = async (mode) => {
    await fetch('/api/start_job', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: mode})
    });
};

window.stopJob = async () => {
    await fetch('/api/stop_job', {method: 'POST'});
};

// Start
updateStatus();
draw();
