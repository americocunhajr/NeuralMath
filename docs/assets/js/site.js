// Optional support link. Set this to a public donation/sponsorship URL when ready.
const SUPPORT_URL = "";
const support = document.querySelector('[data-support]');
if (support) {
  const link = support.querySelector('a');
  if (SUPPORT_URL) {
    link.href = SUPPORT_URL;
    support.hidden = false;
  } else {
    support.hidden = true;
  }
}


const flagBar = document.querySelector('.flags');
const currentFlag = flagBar?.querySelector('.flag.current');
if (flagBar && currentFlag) {
  const target = currentFlag.offsetLeft - (flagBar.clientWidth - currentFlag.clientWidth) / 2;
  flagBar.scrollLeft = Math.max(0, target);
}
