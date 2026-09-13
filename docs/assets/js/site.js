(() => {
  const select = document.querySelector('[data-language-select]');
  if (select) {
    select.addEventListener('change', () => {
      const path = select.options[select.selectedIndex].dataset.path;
      if (path) window.location.href = path;
    });
  }

  document.querySelectorAll('pre').forEach((pre) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block';
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    const toolbar = document.createElement('div');
    toolbar.className = 'code-toolbar';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-btn';
    button.setAttribute('aria-label', 'Copy code');
    button.textContent = document.documentElement.lang.startsWith('pt') ? 'Copiar' :
      document.documentElement.lang.startsWith('es') ? 'Copiar' :
      document.documentElement.lang.startsWith('fr') ? 'Copier' :
      document.documentElement.lang.startsWith('it') ? 'Copia' :
      document.documentElement.lang.startsWith('de') ? 'Kopieren' : 'Copy';

    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        const old = button.textContent;
        button.textContent = '✓';
        setTimeout(() => button.textContent = old, 1200);
      } catch (_) {}
    });
    toolbar.appendChild(button);
    wrapper.insertBefore(toolbar, pre);
  });

  // Highlight the section currently being read on long code pages.
  const tocLinks = [...document.querySelectorAll('.toc a[href^="#"]')];
  const sections = tocLinks
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  if (tocLinks.length && sections.length && 'IntersectionObserver' in window) {
    const map = new Map(tocLinks.map((a) => [a.getAttribute('href').slice(1), a]));
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (!visible) return;
      tocLinks.forEach((a) => a.classList.remove('is-active'));
      const link = map.get(visible.target.id);
      if (link) link.classList.add('is-active');
    }, { rootMargin: '-18% 0px -66% 0px', threshold: 0.01 });
    sections.forEach((section) => observer.observe(section));
  }
})();
