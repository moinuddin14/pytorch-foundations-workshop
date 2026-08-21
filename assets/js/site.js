import { createUISFX } from "uisfx";

(() => {
  const uiSound = createUISFX({
    pack: "zen",
    volume: 0.72,
    preferences: { key: "pytorch-foundations:sound" },
  });
  const soundToggle = document.querySelector("[data-sound-toggle]");
  const soundLabel = soundToggle?.querySelector("[data-sound-label]");
  let hasAudioInteraction = false;

  const playSound = (cue, options) => uiSound.play(cue, options);
  const paintSoundToggle = () => {
    if (!soundToggle || !soundLabel) return;
    const enabled = uiSound.isEnabled();
    soundToggle.setAttribute("aria-pressed", String(enabled));
    soundToggle.setAttribute("aria-label", enabled ? "Mute interface sounds" : "Enable interface sounds");
    soundLabel.textContent = enabled ? "Sound on" : "Sound off";
  };
  const prepareAudio = () => {
    hasAudioInteraction = true;
    if (!uiSound.isEnabled()) return;
    uiSound.unlock().then((unlocked) => {
      if (!unlocked) return;
      uiSound.preload([
        "hover", "open", "close", "forward", "back", "expand", "collapse",
        "copy", "check", "uncheck", "progress-step", "checkpoint", "error",
      ]);
    });
  };

  document.addEventListener("pointerdown", prepareAudio, { capture: true, once: true });
  document.addEventListener("keydown", prepareAudio, { capture: true, once: true });

  paintSoundToggle();
  soundToggle?.addEventListener("click", () => {
    const enabled = !uiSound.isEnabled();
    uiSound.setEnabled(enabled);
    paintSoundToggle();
    if (enabled) {
      uiSound.unlock();
      playSound("toggle-on");
    }
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
  const playNavigation = (event) => {
    const link = event.target.closest("a[href]");
    if (!link || link.classList.contains("skip-link")) return;
    playSound(navigationCue(link));
  };
  document.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    playNavigation(event);
  });
  document.addEventListener("click", (event) => {
    if (event.detail === 0) playNavigation(event);
  });

  const hoverSelector = ".button, .content-card, .path-grid a, .project-grid a, .page-nav a, .site-nav a, .footer-links a";
  document.addEventListener("pointerover", (event) => {
    if (!hasAudioInteraction || event.pointerType === "touch") return;
    const target = event.target.closest(hoverSelector);
    const previous = event.relatedTarget?.closest?.(hoverSelector);
    if (target && target !== previous) playSound("hover", { volume: 0.1 });
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
        playSound(atEnd ? "checkpoint" : "progress-step", { volume: atEnd ? 0.18 : 0.1 });
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
})();
