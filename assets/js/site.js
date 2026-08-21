import { createUISFX } from "uisfx";

(() => {
  const uiSound = createUISFX({
    pack: "soft",
    volume: 0.95,
    preferences: { key: "pytorch-foundations:sound" },
  });
  const soundToggle = document.querySelector("[data-sound-toggle]");
  const soundLabel = soundToggle?.querySelector("[data-sound-label]");
  const cueVolumes = {
    hover: 0.16,
    open: 0.46,
    close: 0.42,
    forward: 0.42,
    back: 0.42,
    seek: 0.32,
    expand: 0.38,
    collapse: 0.34,
    copy: 0.48,
    error: 0.46,
    check: 0.52,
    uncheck: 0.38,
    "progress-step": 0.28,
    checkpoint: 0.48,
    "toggle-on": 0.48,
    "toggle-off": 0.44,
  };
  let hasAudioInteraction = false;
  let audioReady = false;
  let unlockPromise = null;
  let hasPreloaded = false;

  const unlockAudio = () => {
    hasAudioInteraction = true;
    if (!uiSound.isEnabled()) return Promise.resolve(false);
    if (audioReady) return Promise.resolve(true);
    if (!unlockPromise) {
      if (soundToggle) soundToggle.dataset.audioState = "requested";
      unlockPromise = uiSound.unlock().then((unlocked) => {
        audioReady = unlocked;
        if (soundToggle) soundToggle.dataset.audioState = unlocked ? "ready" : "blocked";
        if (!unlocked) unlockPromise = null;
        return unlocked;
      });
    }
    return unlockPromise;
  };
  const playSound = (cue, options = {}) => {
    if (!uiSound.isEnabled()) return null;
    unlockAudio();
    const playback = uiSound.play(cue, { volume: cueVolumes[cue], ...options });
    if (playback && soundToggle) soundToggle.dataset.lastCue = cue;
    return playback;
  };
  const paintSoundToggle = () => {
    if (!soundToggle || !soundLabel) return;
    const enabled = uiSound.isEnabled();
    soundToggle.setAttribute("aria-pressed", String(enabled));
    soundToggle.setAttribute("aria-label", enabled ? "Mute interface sounds" : "Enable interface sounds");
    const pending = soundToggle.dataset.audioState === "requested";
    soundToggle.dataset.audioState = enabled ? (audioReady ? "ready" : (pending ? "requested" : "locked")) : "muted";
    soundLabel.textContent = enabled ? "Sound on" : "Sound off";
  };
  const prepareAudio = () => {
    unlockAudio().then((unlocked) => {
      if (!unlocked || hasPreloaded) return;
      hasPreloaded = true;
      uiSound.preload([
        "hover", "open", "close", "forward", "back", "expand", "collapse",
        "seek", "copy", "check", "uncheck", "progress-step", "checkpoint",
        "error", "toggle-on", "toggle-off",
      ]);
    });
  };

  document.addEventListener("pointerdown", prepareAudio, { capture: true, once: true });
  document.addEventListener("keydown", prepareAudio, { capture: true, once: true });

  paintSoundToggle();
  soundToggle?.addEventListener("click", async () => {
    if (soundToggle.getAttribute("aria-busy") === "true") return;
    soundToggle.setAttribute("aria-busy", "true");
    if (uiSound.isEnabled()) {
      const confirmation = playSound("toggle-off");
      await Promise.race([
        confirmation?.ended ?? Promise.resolve(),
        new Promise((resolve) => window.setTimeout(resolve, 420)),
      ]);
      uiSound.setEnabled(false);
    } else {
      uiSound.setEnabled(true);
      playSound("toggle-on");
    }
    paintSoundToggle();
    soundToggle.removeAttribute("aria-busy");
  });

  const navToggle = document.querySelector("[data-nav-toggle]");
  const nav = document.querySelector("[data-site-nav]");
  if (navToggle && nav) {
    navToggle.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(open));
      navToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
      playSound(open ? "open" : "close");
    });
    nav.addEventListener("click", (event) => {
      if (!event.target.closest("a") || !nav.classList.contains("is-open")) return;
      nav.classList.remove("is-open");
      navToggle.setAttribute("aria-expanded", "false");
      navToggle.setAttribute("aria-label", "Open navigation");
    });
  }

  const navigationCue = (link) => {
    if (link.classList.contains("prev")) return "back";
    if (link.getAttribute("href")?.startsWith("#")) return "seek";
    return "forward";
  };
  let navigationPending = false;
  const playNavigation = (event) => {
    const link = event.target.closest("a[href]");
    if (!link || link.classList.contains("skip-link") || event.defaultPrevented) return;
    const cue = navigationCue(link);
    const destination = new URL(link.href, window.location.href);
    const sameDocument = destination.origin === window.location.origin
      && destination.pathname === window.location.pathname
      && destination.search === window.location.search;
    const modified = event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
    const opensElsewhere = link.target && link.target !== "_self";

    if (sameDocument || modified || opensElsewhere || link.hasAttribute("download")) {
      playSound(cue);
      return;
    }

    event.preventDefault();
    if (navigationPending) return;
    navigationPending = true;
    const playback = playSound(cue);
    window.setTimeout(() => window.location.assign(link.href), playback ? 160 : 0);
  };
  document.addEventListener("click", playNavigation);

  const hoverSelector = ".button, .content-card, .path-grid a, .project-grid a, .page-nav a, .site-nav a, .footer-links a";
  document.addEventListener("pointerover", (event) => {
    if (!hasAudioInteraction || event.pointerType === "touch") return;
    const target = event.target.closest(hoverSelector);
    const previous = event.relatedTarget?.closest?.(hoverSelector);
    if (target && target !== previous) playSound("hover");
  });

  document.querySelectorAll(".article-body pre").forEach((pre) => {
    if (pre.closest(".output") || pre.closest(".code-wrap")) return;
    const code = pre.querySelector("code");
    if (!code) return;
    const wrap = document.createElement("div");
    wrap.className = "code-wrap";
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-code";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied";
        playSound("copy");
        setTimeout(() => { button.textContent = "Copy"; }, 1400);
      } catch (_) {
        button.textContent = "Select";
        playSound("error");
      }
    });
    wrap.appendChild(button);
  });

  const storagePrefix = "pytorch-foundations:complete:";
  const completeButton = document.querySelector("[data-complete-page]");
  if (completeButton) {
    const key = storagePrefix + completeButton.dataset.completePage;
    const paint = (done) => {
      completeButton.classList.toggle("is-complete", done);
      completeButton.innerHTML = done
        ? '<span aria-hidden="true">✓</span> Completed'
        : '<span aria-hidden="true">✓</span> Mark complete';
    };
    paint(localStorage.getItem(key) === "yes");
    completeButton.addEventListener("click", () => {
      const done = localStorage.getItem(key) !== "yes";
      localStorage.setItem(key, done ? "yes" : "no");
      paint(done);
      playSound(done ? "check" : "uncheck");
    });
  }

  document.querySelectorAll("details").forEach((details) => {
    details.addEventListener("toggle", () => {
      playSound(details.open ? "expand" : "collapse");
    });
  });

  document.querySelectorAll("[data-card-path]").forEach((card) => {
    if (localStorage.getItem(storagePrefix + card.dataset.cardPath) === "yes") {
      card.classList.add("is-complete");
      const status = card.querySelector(".card-status");
      if (status) status.textContent = "✓ Completed";
    }
  });

  const progress = document.querySelector(".reading-progress span");
  if (progress) {
    const milestones = [0.25, 0.5, 0.75, 0.99];
    const initialMax = document.documentElement.scrollHeight - window.innerHeight;
    const initialRatio = initialMax > 0 ? window.scrollY / initialMax : 0;
    let nextMilestone = milestones.findIndex((milestone) => milestone > initialRatio);
    if (nextMilestone < 0) nextMilestone = milestones.length;
    let ticking = false;
    const updateProgress = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      progress.style.width = `${ratio * 100}%`;
      if (nextMilestone < milestones.length && ratio >= milestones[nextMilestone]) {
        const atEnd = nextMilestone === milestones.length - 1;
        playSound(atEnd ? "checkpoint" : "progress-step");
        nextMilestone += 1;
      }
      ticking = false;
    };
    updateProgress();
    window.addEventListener("scroll", () => {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(updateProgress);
    }, { passive: true });
  }

  const motionAllowed = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  if (motionAllowed && finePointer) {
    document.querySelectorAll("[data-stellar-hero]").forEach((hero) => {
      const starLayer = hero.querySelector("[data-star-layer]");
      let lastStarX = 0;
      let lastStarY = 0;
      let lastStarTime = 0;

      const spawnStar = (x, y, speed) => {
        if (!starLayer || starLayer.childElementCount > 28) return;
        const star = document.createElement("span");
        const size = Math.min(18, 6 + speed * .26 + Math.random() * 7);
        star.className = "cursor-star";
        star.style.left = `${x + (Math.random() - .5) * 24}px`;
        star.style.top = `${y + (Math.random() - .5) * 24}px`;
        star.style.setProperty("--star-size", `${size}px`);
        star.style.setProperty("--star-rotate", `${Math.random() * 45}deg`);
        star.style.setProperty("--star-drift-x", `${(Math.random() - .5) * 34}px`);
        star.style.setProperty("--star-drift-y", `${-8 - Math.random() * 28}px`);
        star.style.color = Math.random() > .48 ? "#ffffff" : (Math.random() > .5 ? "#8cecff" : "#b6a6ff");
        starLayer.appendChild(star);
        star.addEventListener("animationend", () => star.remove(), { once: true });
      };

      hero.addEventListener("pointerenter", (event) => {
        hero.classList.add("is-pointer-active");
        const rect = hero.getBoundingClientRect();
        lastStarX = event.clientX - rect.left;
        lastStarY = event.clientY - rect.top;
        lastStarTime = performance.now();
        for (let index = 0; index < 4; index += 1) spawnStar(lastStarX, lastStarY, 14);
      });

      hero.addEventListener("pointermove", (event) => {
        const rect = hero.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const percentX = Math.max(0, Math.min(100, x / rect.width * 100));
        const percentY = Math.max(0, Math.min(100, y / rect.height * 100));
        const offsetX = (percentX - 50) * .24;
        const offsetY = (percentY - 50) * .18;
        hero.style.setProperty("--pointer-x", `${percentX}%`);
        hero.style.setProperty("--pointer-y", `${percentY}%`);
        hero.style.setProperty("--parallax-x", `${offsetX}px`);
        hero.style.setProperty("--parallax-y", `${offsetY}px`);
        hero.style.setProperty("--aurora-one", `hsl(${222 + percentX * .34} 86% 59%)`);
        hero.style.setProperty("--aurora-two", `hsl(${170 + percentY * .48} 78% 48%)`);

        const distance = Math.hypot(x - lastStarX, y - lastStarY);
        const now = performance.now();
        if (distance > 13 && now - lastStarTime > 30) {
          spawnStar(x, y, distance);
          lastStarX = x;
          lastStarY = y;
          lastStarTime = now;
        }
      }, { passive: true });

      hero.addEventListener("pointerleave", () => {
        hero.classList.remove("is-pointer-active");
        hero.style.setProperty("--pointer-x", "72%");
        hero.style.setProperty("--pointer-y", "28%");
        hero.style.setProperty("--parallax-x", "0px");
        hero.style.setProperty("--parallax-y", "0px");
        hero.style.removeProperty("--aurora-one");
        hero.style.removeProperty("--aurora-two");
      });
    });
  }
})();
