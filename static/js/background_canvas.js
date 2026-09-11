/**
 * ═══════════════════════════════════════════════════════════════════════════
 * F-INTEL QUANTITATIVE TERMINAL — ANIMATED QUANT LATTICE BACKGROUND
 * ═══════════════════════════════════════════════════════════════════════════
 * 60 FPS hardware-accelerated dynamic particle constellation & volatility grid.
 * Features:
 *   - Auto-throttling via requestAnimationFrame
 *   - Visibility-aware pausing (0% CPU when tab is hidden)
 *   - Gentle mouse proximity physics
 *   - Dynamic density and crisp DPI scaling
 */

(function () {
    'use strict';

    var canvas = document.getElementById('quant-bg-canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'quant-bg-canvas';
        document.body.prepend(canvas);
    }

    var ctx = canvas.getContext('2d');
    var animationFrameId = null;
    var isEnabled = true;
    var isVisible = true;

    // Grid particles count (symbolic 65 for NIFTY lot size)
    var PARTICLE_COUNT = 65;
    var CONNECT_DISTANCE = 135;
    var MOUSE_RADIUS = 120;

    var width = 0;
    var height = 0;
    var dpr = window.devicePixelRatio || 1;

    var particles = [];
    var mouse = { x: null, y: null };

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        ctx.scale(dpr, dpr);
    }

    function Particle() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.45;
        this.vy = (Math.random() - 0.5) * 0.45;
        this.radius = Math.random() * 1.6 + 0.8;
        this.baseAlpha = Math.random() * 0.35 + 0.15;
        this.alpha = this.baseAlpha;
        this.pulseSpeed = Math.random() * 0.02 + 0.01;
        this.pulseAngle = Math.random() * Math.PI * 2;
    }

    Particle.prototype.update = function () {
        this.x += this.vx;
        this.y += this.vy;

        // Screen boundary wrap
        if (this.x < -10) this.x = width + 10;
        if (this.x > width + 10) this.x = -10;
        if (this.y < -10) this.y = height + 10;
        if (this.y > height + 10) this.y = -10;

        // Subtle alpha pulsing
        this.pulseAngle += this.pulseSpeed;
        this.alpha = this.baseAlpha + Math.sin(this.pulseAngle) * 0.1;

        // Mouse proximity interaction
        if (mouse.x !== null && mouse.y !== null) {
            var dx = mouse.x - this.x;
            var dy = mouse.y - this.y;
            var dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < MOUSE_RADIUS) {
                var force = (1 - dist / MOUSE_RADIUS) * 0.8;
                this.x -= (dx / dist) * force;
                this.y -= (dy / dist) * force;
            }
        }
    };

    Particle.prototype.draw = function () {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 240, 255, ' + Math.max(this.alpha, 0.05) + ')';
        ctx.fill();
    };

    function initParticles() {
        particles = [];
        for (var i = 0; i < PARTICLE_COUNT; i++) {
            particles.push(new Particle());
        }
    }

    function render() {
        if (!isEnabled || !isVisible) {
            animationFrameId = null;
            return;
        }

        ctx.clearRect(0, 0, width, height);

        // Draw connecting lattice lines
        var pLen = particles.length;
        for (var i = 0; i < pLen; i++) {
            var p1 = particles[i];
            p1.update();
            p1.draw();

            for (var j = i + 1; j < pLen; j++) {
                var p2 = particles[j];
                var dx = p1.x - p2.x;
                var dy = p1.y - p2.y;
                var dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < CONNECT_DISTANCE) {
                    var lineAlpha = (1 - dist / CONNECT_DISTANCE) * 0.12;
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = 'rgba(0, 240, 255, ' + lineAlpha + ')';
                    ctx.lineWidth = 0.75;
                    ctx.stroke();
                }
            }
        }

        animationFrameId = requestAnimationFrame(render);
    }

    // Event Listeners
    window.addEventListener('resize', function () {
        resize();
        initParticles();
    });

    window.addEventListener('mousemove', function (e) {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });

    window.addEventListener('mouseleave', function () {
        mouse.x = null;
        mouse.y = null;
    });

    document.addEventListener('visibilitychange', function () {
        isVisible = !document.hidden;
        if (isVisible && isEnabled && !animationFrameId) {
            render();
        }
    });

    // Public API for header Ambient FX toggle switch
    window.toggleAmbientFX = function (state) {
        if (typeof state === 'boolean') {
            isEnabled = state;
        } else {
            isEnabled = !isEnabled;
        }
        canvas.style.opacity = isEnabled ? '0.85' : '0.0';
        if (isEnabled && !animationFrameId) {
            render();
        }
        try {
            localStorage.setItem('fintel_ambient_fx', isEnabled ? '1' : '0');
        } catch (e) {}
        return isEnabled;
    };

    // Initialize
    resize();
    initParticles();
    
    // Check saved preference
    try {
        var savedPref = localStorage.getItem('fintel_ambient_fx');
        if (savedPref === '0') {
            window.toggleAmbientFX(false);
        } else {
            render();
        }
    } catch (e) {
        render();
    }
})();
