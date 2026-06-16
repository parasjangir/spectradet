// ============ nav scroll state ============
const nav = document.getElementById("nav");
const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 20);
onScroll();
window.addEventListener("scroll", onScroll, { passive: true });

// ============ mobile menu ============
const burger = document.getElementById("burger");
const links = document.querySelector(".nav__links");
burger?.addEventListener("click", () => links.classList.toggle("open"));
links?.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => links.classList.remove("open")));

// ============ cursor glow ============
const glow = document.querySelector(".cursor-glow");
if (matchMedia("(pointer:fine)").matches) {
  window.addEventListener("mousemove", (e) => {
    glow.style.opacity = "1";
    glow.style.left = e.clientX + "px";
    glow.style.top = e.clientY + "px";
  });
}

// ============ feature-card spotlight ============
document.querySelectorAll(".feat").forEach((card) => {
  card.addEventListener("mousemove", (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  });
});

// ============ scroll reveal + counters ============
const animateCount = (el) => {
  const to = parseFloat(el.dataset.to);
  const dec = parseInt(el.dataset.dec || "0", 10);
  const dur = 1400;
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
    el.textContent = (to * eased).toFixed(dec);
    if (p < 1) requestAnimationFrame(tick);
    else el.textContent = to.toFixed(dec);
  };
  requestAnimationFrame(tick);
};

const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      el.classList.add("in");
      el.querySelectorAll?.(".count").forEach((c) => {
        if (!c.dataset.done) { c.dataset.done = "1"; animateCount(c); }
      });
      io.unobserve(el);
    });
  },
  { threshold: 0.18 }
);
document.querySelectorAll("[data-reveal]").forEach((el, i) => {
  el.style.transitionDelay = `${(i % 4) * 70}ms`;
  io.observe(el);
});

// ============ hero FFT-wave canvas ============
const canvas = document.getElementById("fft-canvas");
if (canvas && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
  const ctx = canvas.getContext("2d");
  let w, h, dpr;
  const resize = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.width = canvas.offsetWidth * dpr;
    h = canvas.height = canvas.offsetHeight * dpr;
  };
  resize();
  window.addEventListener("resize", resize);

  const waves = [
    { amp: 0.10, freq: 1.4, speed: 0.6, color: "rgba(34,211,238,0.55)" },
    { amp: 0.07, freq: 2.7, speed: -0.9, color: "rgba(168,85,247,0.5)" },
    { amp: 0.05, freq: 4.6, speed: 1.3, color: "rgba(236,72,153,0.45)" },
  ];

  const draw = (t) => {
    ctx.clearRect(0, 0, w, h);
    const mid = h * 0.62;
    waves.forEach((wv, idx) => {
      ctx.beginPath();
      for (let x = 0; x <= w; x += 6 * dpr) {
        const k = x / w;
        // sum of sinusoids modulated by a moving gaussian envelope = "spectrum" feel
        const env = Math.exp(-Math.pow((k - 0.5) * 2.2, 2));
        const y =
          mid +
          Math.sin(k * Math.PI * 2 * wv.freq + t * 0.001 * wv.speed * 6) *
            h * wv.amp * env +
          Math.sin(k * Math.PI * 2 * (wv.freq * 2.3) + t * 0.0013 * wv.speed * 4) *
            h * wv.amp * 0.4 * env;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = wv.color;
      ctx.lineWidth = 1.6 * dpr;
      ctx.stroke();
    });
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}

// ============ active nav link on scroll ============
const sections = [...document.querySelectorAll("section[id]")];
const navLinks = [...document.querySelectorAll('.nav__links a')];
const spy = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        navLinks.forEach((l) =>
          l.style.setProperty("color", l.getAttribute("href") === `#${e.target.id}` ? "var(--text)" : "")
        );
      }
    });
  },
  { rootMargin: "-40% 0px -55% 0px" }
);
sections.forEach((s) => spy.observe(s));
