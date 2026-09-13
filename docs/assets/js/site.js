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
