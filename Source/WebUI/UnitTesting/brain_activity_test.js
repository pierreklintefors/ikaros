const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");


// The widget file also defines the WebUI element; stub what it extends and registers with.
const widgetPath = path.resolve(__dirname, "../widgets/WebUIWidgetBrainActivity.js");
const context = vm.createContext({
    Math,
    console,
    WebUIWidgetCanvas: class {},
    webui_widgets: { add() {} },
});
vm.runInContext(
    fs.readFileSync(widgetPath, "utf8") + "\nglobalThis.RendererUnderTest = BrainActivityRenderer;\n",
    context,
    { filename: widgetPath },
);
const Renderer = context.RendererUnderTest;


function run(renderer, seconds, targets, pupil)
{
    for(let t = 0; t < seconds; t += 1 / 60)
        renderer.step(1 / 60, targets, pupil, { max: 10, speed: 420 });
    const counts = {};
    for(const p of renderer.paths)
        counts[`${p.from}>${p.to}`] = p.sent;
    return counts;
}


// Every pathway starts and ends at a known node or the iris.
const renderer = new Renderer();
for(const p of renderer.paths)
{
    assert.ok(renderer.nodes[p.from], `unknown pathway start ${p.from}`);
    assert.ok(p.to == "iris" || renderer.nodes[p.to], `unknown pathway end ${p.to}`);
    assert.ok(p.length > 0);
}

// Silent nuclei send no spikes.
const silent = run(new Renderer(), 3, {}, 0.5);
assert.equal(Object.values(silent).reduce((a, b) => a + b, 0), 0);

// An active light reflex sends spikes along its own pathways only.
const lightOnly = { light: 1, pta: 1, ew: 1, cg: 1 };
const light = run(new Renderer(), 3, lightOnly, 0.5);
assert.ok(light["light>pta"] > 10, `light>pta ${light["light>pta"]}`);
assert.ok(light["cg>iris"] > 10, `cg>iris ${light["cg>iris"]}`);
assert.equal(light["lc>iml"], 0);
assert.equal(light["amygdala>lc"], 0);

// More activity gives more spikes.
const weak = run(new Renderer(), 3, { lc: 0.4 }, 0.5);
const strong = run(new Renderer(), 3, { lc: 1 }, 0.5);
assert.ok(strong["lc>iml"] > weak["lc>iml"] * 2, `weak ${weak["lc>iml"]} strong ${strong["lc>iml"]}`);

// Displayed values and the pupil approach their targets.
const follow = new Renderer();
run(follow, 2, { amygdala: 0.8 }, 0.9);
assert.ok(Math.abs(follow.values.amygdala - 0.8) < 0.01);
assert.ok(Math.abs(follow.pupil - 0.9) < 0.01);

// Rendering uses only standard 2D context calls; a recording stub must not throw.
const calls = new Set();
const gradient = { addColorStop() {} };
const g = new Proxy({}, {
    get(target, name)
    {
        if(name in target)
            return target[name];
        if(name == "measureText")
            return () => ({ width: 40 });
        if(name == "createRadialGradient" || name == "createLinearGradient")
            return () => gradient;
        return (...args) => { calls.add(name); };
    },
    set(target, name, value)
    {
        target[name] = value;
        return true;
    },
});
follow.render(g, 800, 500, true);
for(const name of ["arc", "fill", "stroke", "fillText", "quadraticCurveTo", "bezierCurveTo"])
    assert.ok(calls.has(name), `render did not call ${name}`);

console.log("brain activity renderer tests passed");
