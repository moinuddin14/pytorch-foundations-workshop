(() => {
  const navToggle = document.querySelector("[data-nav-toggle]");
  const nav = document.querySelector("[data-site-nav]");
  if (navToggle && nav) {
    navToggle.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
    nav.addEventListener("click", () => {
      nav.classList.remove("is-open");
      navToggle.setAttribute("aria-expanded", "false");
    });
  }

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
        setTimeout(() => { button.textContent = "Copy"; }, 1400);
      } catch (_) {
        button.textContent = "Select";
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
    });
  }

  document.querySelectorAll("[data-card-path]").forEach((card) => {
    if (localStorage.getItem(storagePrefix + card.dataset.cardPath) === "yes") {
      card.classList.add("is-complete");
      const status = card.querySelector(".card-status");
      if (status) status.textContent = "✓ Completed";
    }
  });

  const progress = document.querySelector(".reading-progress span");
  if (progress) {
    const updateProgress = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      progress.style.width = `${max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0}%`;
    };
    updateProgress();
    window.addEventListener("scroll", updateProgress, { passive: true });
  }
})();
