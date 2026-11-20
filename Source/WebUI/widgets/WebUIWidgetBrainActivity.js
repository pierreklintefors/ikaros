// Animated side view of the pupil brain: nuclei glow with their activity and spikes travel along
// the pathways between them. The drawing code only needs a 2D context, so it can be tried outside
// the WebUI with BrainActivityRenderer.

class BrainActivityRenderer
{
    constructor()
    {
        // Virtual drawing space; the renderer scales it to fit the canvas.
        this.W = 1000;
        this.H = 620;

        // kind: light (yellow), para (parasympathetic, cyan), symp (sympathetic, orange),
        // emo (emotion, magenta), sense (touch, pink)
        this.nodes = {
            light:    { x: 178, y: 300, label: "Retina", note: "light", kind: "light" },
            pta:      { x: 520, y: 214, label: "PTA", note: "pretectal area", kind: "para" },
            ew:       { x: 590, y: 286, label: "EW", note: "Edinger-Westphal", kind: "para" },
            cg:       { x: 236, y: 352, label: "CG", note: "ciliary ganglion", kind: "para" },
            visual:   { x: 838, y: 214, label: "V1", note: "sees the cue", kind: "emo" },
            touch:    { x: 58,  y: 166, label: "Poke", note: "finger close", kind: "sense" },
            amygdala: { x: 452, y: 378, label: "Amygdala", note: "fear learning", kind: "emo" },
            arousal:  { x: 372, y: 256, label: "Arousal", note: "hypothalamus", kind: "symp" },
            lc:       { x: 616, y: 392, label: "LC", note: "locus coeruleus", kind: "symp" },
            iml:      { x: 640, y: 560, label: "IML", note: "spinal cord", kind: "symp" },
            scg:      { x: 384, y: 540, label: "SCG", note: "superior cervical", kind: "symp" },
        };

        // Curved pathways: from, to, control point, kind, and whether they inhibit.
        this.paths = [
            { from: "light", to: "pta", c: [330, 372], kind: "light" },
            { from: "light", to: "visual", c: [500, 100], kind: "light", faint: true },
            { from: "pta", to: "ew", c: [580, 230], kind: "para" },
            { from: "ew", to: "cg", c: [430, 320], kind: "para" },
            { from: "cg", to: "iris", c: [190, 330], kind: "para" },
            { from: "visual", to: "amygdala", c: [700, 250], kind: "emo" },
            { from: "touch", to: "amygdala", c: [250, 220], kind: "sense" },
            { from: "amygdala", to: "lc", c: [530, 340], kind: "emo" },
            { from: "arousal", to: "lc", c: [480, 440], kind: "symp" },
            { from: "lc", to: "ew", c: [650, 330], kind: "inhib", inhibit: true },
            { from: "lc", to: "iml", c: [650, 470], kind: "symp" },
            { from: "iml", to: "scg", c: [520, 600], kind: "symp" },
            { from: "scg", to: "iris", c: [200, 470], kind: "symp" },
        ];

        this.colors = {
            light: [255, 214, 80],
            para:  [70, 220, 255],
            symp:  [255, 140, 50],
            emo:   [235, 80, 255],
            sense: [255, 110, 170],
            inhib: [255, 60, 80],
        };

        this.eye = { x: 104, y: 276, r: 50 };
        for(const p of this.paths)
        {
            p.a = this.nodes[p.from];
            p.b = p.to == "iris" ? { x: this.eye.x + 18, y: this.eye.y } : this.nodes[p.to];
            p.length = this.curveLength(p);
            p.spikes = [];
            p.sent = 0;
            p.emit = Math.random();
        }

        this.values = {};
        this.flash = {};
        for(const k of Object.keys(this.nodes))
        {
            this.values[k] = 0;
            this.flash[k] = 0;
        }
        this.pupil = 0.5;
        this.time = 0;

        // Background specks that drift slowly, like distant neural activity.
        this.specks = [];
        for(let i = 0; i < 90; i++)
            this.specks.push({ x: Math.random() * this.W, y: Math.random() * this.H,
                               p: Math.random() * Math.PI * 2, s: 0.4 + Math.random() * 1.2 });
    }


    point(p, t)
    {
        const u = 1 - t;
        return {
            x: u * u * p.a.x + 2 * u * t * p.c[0] + t * t * p.b.x,
            y: u * u * p.a.y + 2 * u * t * p.c[1] + t * t * p.b.y,
        };
    }


    curveLength(p)
    {
        let length = 0;
        let last = this.point(p, 0);
        for(let i = 1; i <= 24; i++)
        {
            const q = this.point(p, i / 24);
            length += Math.hypot(q.x - last.x, q.y - last.y);
            last = q;
        }
        return length;
    }


    rgba(c, a)
    {
        return `rgba(${c[0]},${c[1]},${c[2]},${Math.max(0, Math.min(1, a))})`;
    }


    // targets: activity per node from 0 to about 1, and pupil from 0 (small) to 1 (large)
    step(dt, targets, pupil, rates)
    {
        this.time += dt;
        const follow = 1 - Math.exp(-dt / 0.12);
        for(const k of Object.keys(this.nodes))
        {
            const target = Number.isFinite(targets[k]) ? targets[k] : 0;
            this.values[k] += follow * (target - this.values[k]);
            this.flash[k] *= Math.exp(-dt / 0.18);
        }
        if(Number.isFinite(pupil))
            this.pupil += (1 - Math.exp(-dt / 0.2)) * (pupil - this.pupil);

        const speed = rates.speed;
        for(const p of this.paths)
        {
            // Spike rate grows with the activity of the sending nucleus.
            const activity = Math.max(0, Math.min(1.5, this.values[p.from]));
            p.emit += dt * rates.max * activity * activity * (p.faint ? 0.35 : 1);
            while(p.emit >= 1)
            {
                p.emit -= 1 + 0.6 * (Math.random() - 0.5);
                p.sent++;
                if(p.spikes.length < 24)
                    p.spikes.push({ t: 0, w: 0.7 + 0.6 * Math.random() });
            }
            for(const s of p.spikes)
                s.t += dt * speed / p.length;
            while(p.spikes.length && p.spikes[0].t >= 1)
            {
                p.spikes.shift();
                if(p.to != "iris")
                    this.flash[p.to] = Math.min(1, this.flash[p.to] + (p.inhibit ? 0.25 : 0.35));
            }
        }
    }


    drawAnatomy(g)
    {
        // Cortex
        g.save();
        g.beginPath();
        g.moveTo(250, 250);
        g.bezierCurveTo(240, 120, 380, 40, 540, 46);
        g.bezierCurveTo(720, 50, 880, 110, 900, 230);
        g.bezierCurveTo(912, 300, 880, 340, 820, 330);
        g.bezierCurveTo(760, 318, 700, 330, 660, 340);
        g.bezierCurveTo(620, 350, 600, 330, 560, 334);
        g.bezierCurveTo(500, 340, 470, 400, 400, 396);
        g.bezierCurveTo(330, 392, 300, 330, 250, 250);
        g.closePath();
        const cortex = g.createLinearGradient(250, 40, 880, 400);
        cortex.addColorStop(0, "rgba(60,90,170,0.20)");
        cortex.addColorStop(1, "rgba(120,60,170,0.16)");
        g.fillStyle = cortex;
        g.fill();
        g.lineWidth = 2;
        g.strokeStyle = "rgba(140,170,255,0.35)";
        g.stroke();

        // Gyri: soft curved strokes inside the cortex
        g.strokeStyle = "rgba(140,170,255,0.12)";
        g.lineWidth = 1.5;
        const gyri = [
            [[330, 140], [380, 100], [430, 150]], [[440, 90], [500, 70], [530, 130]],
            [[560, 80], [620, 110], [600, 170]], [[650, 110], [720, 120], [700, 190]],
            [[740, 150], [820, 170], [790, 250]], [[340, 230], [400, 200], [450, 240]],
            [[480, 160], [560, 180], [620, 230]], [[660, 230], [740, 230], [780, 290]],
        ];
        for(const [a, c, b] of gyri)
        {
            g.beginPath();
            g.moveTo(a[0], a[1]);
            g.quadraticCurveTo(c[0], c[1], b[0], b[1]);
            g.stroke();
        }

        // Cerebellum
        g.beginPath();
        g.ellipse(770, 380, 96, 54, -0.15, 0, Math.PI * 2);
        g.fillStyle = "rgba(90,70,160,0.20)";
        g.fill();
        g.strokeStyle = "rgba(160,140,255,0.30)";
        g.lineWidth = 2;
        g.stroke();
        g.strokeStyle = "rgba(160,140,255,0.12)";
        for(let i = 1; i < 5; i++)
        {
            g.beginPath();
            g.ellipse(770, 380, 96 - i * 16, 54 - i * 9, -0.15, Math.PI * 1.05, Math.PI * 1.95);
            g.stroke();
        }

        // Brainstem and spinal cord
        g.beginPath();
        g.moveTo(560, 330);
        g.bezierCurveTo(600, 400, 620, 470, 626, 620);
        g.lineTo(666, 620);
        g.bezierCurveTo(660, 470, 650, 390, 640, 336);
        g.closePath();
        g.fillStyle = "rgba(80,120,190,0.18)";
        g.fill();
        g.strokeStyle = "rgba(140,170,255,0.30)";
        g.lineWidth = 1.5;
        g.stroke();
        g.restore();
    }


    drawEye(g)
    {
        const e = this.eye;
        g.save();
        // Sclera
        const sclera = g.createRadialGradient(e.x - 12, e.y - 14, 4, e.x, e.y, e.r);
        sclera.addColorStop(0, "rgba(245,248,255,0.95)");
        sclera.addColorStop(1, "rgba(150,165,200,0.85)");
        g.beginPath();
        g.arc(e.x, e.y, e.r, 0, Math.PI * 2);
        g.fillStyle = sclera;
        g.fill();

        // Iris looking left, out of the head
        const ix = e.x - 20;
        const ir = 30;
        const sym = this.values.scg;
        const para = this.values.cg;
        const iris = g.createRadialGradient(ix, e.y, 6, ix, e.y, ir);
        iris.addColorStop(0, this.rgba([70 + 120 * para, 150, 210], 1));
        iris.addColorStop(1, this.rgba([30, 60 + 40 * sym, 120], 1));
        g.beginPath();
        g.arc(ix, e.y, ir, 0, Math.PI * 2);
        g.fillStyle = iris;
        g.fill();

        // Iris fibres
        g.strokeStyle = "rgba(255,255,255,0.18)";
        g.lineWidth = 1;
        for(let i = 0; i < 24; i++)
        {
            const a = i / 24 * Math.PI * 2;
            g.beginPath();
            g.moveTo(ix + Math.cos(a) * ir * 0.45, e.y + Math.sin(a) * ir * 0.45);
            g.lineTo(ix + Math.cos(a) * ir * 0.95, e.y + Math.sin(a) * ir * 0.95);
            g.stroke();
        }

        // Pupil
        const pr = ir * (0.22 + 0.62 * Math.max(0, Math.min(1, this.pupil)));
        g.beginPath();
        g.arc(ix, e.y, pr, 0, Math.PI * 2);
        g.fillStyle = "#03040a";
        g.fill();
        g.beginPath();
        g.arc(ix - pr * 0.35, e.y - pr * 0.4, Math.max(2, pr * 0.22), 0, Math.PI * 2);
        g.fillStyle = "rgba(255,255,255,0.75)";
        g.fill();
        g.restore();
    }


    drawPaths(g)
    {
        g.save();
        g.lineCap = "round";
        for(const p of this.paths)
        {
            const c = this.colors[p.kind];
            const activity = Math.min(1, this.values[p.from]);
            g.beginPath();
            g.moveTo(p.a.x, p.a.y);
            g.quadraticCurveTo(p.c[0], p.c[1], p.b.x, p.b.y);
            g.setLineDash(p.inhibit ? [6, 6] : []);
            g.lineWidth = 2 + 3 * activity;
            g.strokeStyle = this.rgba(c, (p.faint ? 0.06 : 0.12) + 0.25 * activity);
            g.stroke();

            if(p.inhibit)
            {
                // Bar at the end of an inhibitory pathway
                const q = this.point(p, 0.97);
                const r = this.point(p, 0.9);
                const a = Math.atan2(q.y - r.y, q.x - r.x) + Math.PI / 2;
                g.setLineDash([]);
                g.lineWidth = 3;
                g.strokeStyle = this.rgba(c, 0.4 + 0.5 * activity);
                g.beginPath();
                g.moveTo(q.x + 9 * Math.cos(a), q.y + 9 * Math.sin(a));
                g.lineTo(q.x - 9 * Math.cos(a), q.y - 9 * Math.sin(a));
                g.stroke();
            }
        }
        g.setLineDash([]);
        g.restore();
    }


    drawSpikes(g)
    {
        g.save();
        g.globalCompositeOperation = "lighter";
        for(const p of this.paths)
        {
            const c = this.colors[p.kind];
            for(const s of p.spikes)
            {
                // A short comet: a bright head and a fading trail along the curve
                for(let k = 5; k >= 0; k--)
                {
                    const t = s.t - k * 0.018;
                    if(t < 0)
                        continue;
                    const q = this.point(p, t);
                    const fade = 1 - k / 6;
                    const r = (2.2 + 2.8 * fade) * s.w;
                    g.beginPath();
                    g.arc(q.x, q.y, r, 0, Math.PI * 2);
                    g.fillStyle = this.rgba(c, 0.55 * fade);
                    g.fill();
                }
                const head = this.point(p, s.t);
                const glow = g.createRadialGradient(head.x, head.y, 0, head.x, head.y, 16 * s.w);
                glow.addColorStop(0, this.rgba([255, 255, 255], 0.9));
                glow.addColorStop(0.25, this.rgba(c, 0.6));
                glow.addColorStop(1, this.rgba(c, 0));
                g.fillStyle = glow;
                g.beginPath();
                g.arc(head.x, head.y, 16 * s.w, 0, Math.PI * 2);
                g.fill();
            }
        }
        g.restore();
    }


    drawNodes(g, showValues)
    {
        g.save();
        for(const [k, n] of Object.entries(this.nodes))
        {
            const c = this.colors[n.kind];
            const v = Math.max(0, Math.min(1.5, this.values[k]));
            const f = this.flash[k];
            const level = Math.min(1, 0.65 * v + 0.6 * f);
            const r = 9 + 9 * Math.min(1, v) + 5 * f;

            // Halo
            g.globalCompositeOperation = "lighter";
            const halo = g.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 4);
            halo.addColorStop(0, this.rgba(c, 0.15 + 0.6 * level));
            halo.addColorStop(0.4, this.rgba(c, 0.08 + 0.25 * level));
            halo.addColorStop(1, this.rgba(c, 0));
            g.fillStyle = halo;
            g.beginPath();
            g.arc(n.x, n.y, r * 4, 0, Math.PI * 2);
            g.fill();

            // Pulse rings, faster with more activity
            const period = 1.6 - 1.1 * Math.min(1, v);
            for(let i = 0; i < 2; i++)
            {
                const phase = ((this.time / period) + i * 0.5) % 1;
                g.beginPath();
                g.arc(n.x, n.y, r + phase * r * 2.2, 0, Math.PI * 2);
                g.lineWidth = 1.5;
                g.strokeStyle = this.rgba(c, (1 - phase) * 0.5 * level);
                g.stroke();
            }

            // Core
            g.globalCompositeOperation = "source-over";
            const core = g.createRadialGradient(n.x - r * 0.3, n.y - r * 0.3, 1, n.x, n.y, r);
            core.addColorStop(0, this.rgba([255, 255, 255], 0.5 + 0.5 * level));
            core.addColorStop(0.45, this.rgba(c, 0.55 + 0.45 * level));
            core.addColorStop(1, this.rgba(c.map(x => x * 0.35), 0.9));
            g.fillStyle = core;
            g.beginPath();
            g.arc(n.x, n.y, r, 0, Math.PI * 2);
            g.fill();

            // Labels
            g.font = "bold 15px -apple-system, Helvetica, sans-serif";
            g.textAlign = "center";
            g.fillStyle = this.rgba([235, 240, 255], 0.75 + 0.25 * level);
            g.fillText(n.label, n.x, n.y - r - 8);
            g.font = "11px -apple-system, Helvetica, sans-serif";
            g.fillStyle = "rgba(190,200,230,0.65)";
            const note = showValues ? `${n.note} ${this.values[k].toFixed(2)}` : n.note;
            g.fillText(note, n.x, n.y + r + 15);
        }
        g.restore();
    }


    drawBackground(g)
    {
        const bg = g.createRadialGradient(500, 280, 60, 500, 310, 640);
        bg.addColorStop(0, "#101a36");
        bg.addColorStop(1, "#03050d");
        g.fillStyle = bg;
        g.fillRect(0, 0, this.W, this.H);

        // Overall activity lights up the background a little
        const total = (this.values.lc + this.values.amygdala + this.values.pta) / 3;
        g.save();
        g.globalCompositeOperation = "lighter";
        for(const s of this.specks)
        {
            const tw = 0.5 + 0.5 * Math.sin(this.time * s.s * 2 + s.p);
            g.fillStyle = `rgba(150,180,255,${0.05 + 0.18 * tw * (0.4 + total)})`;
            g.fillRect(s.x, s.y, 1.6, 1.6);
        }
        g.restore();
    }


    drawLegend(g)
    {
        const items = [
            ["light", "Light"], ["para", "Narrows the pupil"], ["symp", "Widens the pupil"],
            ["emo", "Emotion"], ["inhib", "Inhibition"],
        ];
        g.save();
        g.font = "12px -apple-system, Helvetica, sans-serif";
        g.textAlign = "left";
        let x = 24;
        for(const [kind, text] of items)
        {
            g.fillStyle = this.rgba(this.colors[kind], 0.95);
            g.beginPath();
            g.arc(x + 5, this.H - 22, 5, 0, Math.PI * 2);
            g.fill();
            g.fillStyle = "rgba(210,220,245,0.8)";
            g.fillText(text, x + 15, this.H - 18);
            x += 34 + g.measureText(text).width;
        }
        g.restore();
    }


    // g: 2D context with its transform reset; width and height in CSS pixels.
    render(g, width, height, showValues)
    {
        const s = Math.min(width / this.W, height / this.H);
        g.save();
        g.fillStyle = "#03050d";
        g.fillRect(0, 0, width, height);
        g.translate((width - this.W * s) / 2, (height - this.H * s) / 2);
        g.scale(s, s);
        this.drawBackground(g);
        this.drawAnatomy(g);
        this.drawPaths(g);
        this.drawEye(g);
        this.drawSpikes(g);
        this.drawNodes(g, showValues);
        this.drawLegend(g);
        g.restore();
    }
}


class WebUIWidgetBrainActivity extends WebUIWidgetCanvas
{
    static template()
    {
        return [
            { name: "SOURCE", control: "header" },
            { name: "title", default: "Brain activity", type: "string", control: "textedit" },
            { name: "light_source", default: "", type: "source", control: "textedit" },
            { name: "pta_source", default: "", type: "source", control: "textedit" },
            { name: "ew_source", default: "", type: "source", control: "textedit" },
            { name: "cg_source", default: "", type: "source", control: "textedit" },
            { name: "visual_source", default: "", type: "source", control: "textedit" },
            { name: "touch_source", default: "", type: "source", control: "textedit" },
            { name: "amygdala_source", default: "", type: "source", control: "textedit" },
            { name: "arousal_source", default: "", type: "source", control: "textedit" },
            { name: "lc_source", default: "", type: "source", control: "textedit" },
            { name: "iml_source", default: "", type: "source", control: "textedit" },
            { name: "scg_source", default: "", type: "source", control: "textedit" },
            { name: "pupil_source", default: "", type: "source", control: "textedit" },
            { name: "PARAMETERS", control: "header" },
            { name: "pupil_min", default: 5, type: "float", control: "textedit" },
            { name: "pupil_max", default: 16, type: "float", control: "textedit" },
            { name: "max_activity", default: 1.2, type: "float", control: "textedit" },
            { name: "spike_rate", default: 10, type: "float", control: "textedit" },
            { name: "show_values", default: false, type: "bool", control: "checkbox" },
        ];
    }


    init()
    {
        super.init();
        this.renderer = new BrainActivityRenderer();
        this.targets = {};
        this.pupilTarget = 0.5;
        this.lastFrame = undefined;
        this.running = true;
        const frame = (now) =>
        {
            if(!this.running)
                return;
            if(!document.hidden && this.isConnected && this.width > 1)
            {
                const dt = this.lastFrame === undefined ? 0 : Math.min(0.1, (now - this.lastFrame) / 1000);
                this.renderer.step(dt, this.targets, this.pupilTarget,
                                   { max: this.finiteNumber(this.parameters.spike_rate, 10), speed: 420 });
                this.resetCanvasTransform(0, 0);
                this.renderer.render(this.canvas, this.width, this.height, this.toBool(this.parameters.show_values));
            }
            this.lastFrame = now;
            requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
        this.updateAll();
    }


    disconnectedCallback()
    {
        this.running = false;
        super.disconnectedCallback();
    }


    update()
    {
        if(!this.renderer)
            return;
        const scale = this.positiveNumber(this.parameters.max_activity, 1.2);
        for(const k of Object.keys(this.renderer.nodes))
            this.targets[k] = this.sourceNumber(`${k}_source`, 0) / scale;
        const min = this.finiteNumber(this.parameters.pupil_min, 5);
        const max = this.finiteNumber(this.parameters.pupil_max, 16);
        const pupil = this.sourceNumber("pupil_source", NaN);
        if(Number.isFinite(pupil) && max > min)
            this.pupilTarget = (pupil - min) / (max - min);
    }
}


webui_widgets.add("webui-widget-brain-activity", WebUIWidgetBrainActivity);
