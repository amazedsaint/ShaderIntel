// A small syntax highlighter for the site: JavaScript / JSX, WGSL, HTML (with inline scripts) and
// shell. One tokenising regex per language; the output is escaped HTML with span classes.

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const span = (cls, s) => `<span class="hl-${cls}">${esc(s)}</span>`;

const JS_KW = new Set(('await async break case catch class const continue default delete do else export extends finally for from function if import in ' +
  'instanceof let new of return static super switch this throw try typeof var void while yield true false null undefined').split(' '));
const WGSL_KW = new Set(('fn let var const struct return if else for loop while break continue switch case default discard true false enable requires ' +
  'override alias select array atomic ptr workgroup storage uniform read read_write private function').split(' '));
const WGSL_TYPES = /^(?:f32|f16|i32|u32|bool|vec[234][fiuh]?|mat[234]x[234][fh]?|texture_\w+|sampler)$/;

function code(src, keywords, extra = '') {
  const re = new RegExp(String.raw`(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|('(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"|\`(?:\\.|[^\`\\])*\`)|(\b0x[0-9a-fA-F]+u?\b|\b\d[\d_]*(?:\.\d+)?(?:e[-+]?\d+)?[fuih]?\b)|(@\w+|\b[A-Za-z_$][\w$]*\b)${extra}`, 'g');
  let out = '', last = 0, m;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    last = re.lastIndex;
    const [tok, com, str, num, word] = m;
    if (com) out += span('c', com);
    else if (str) out += str.startsWith('`') ? template(str) : span('s', str);
    else if (num) out += span('n', num);
    else if (word) {
      if (keywords.has(word)) out += span('k', word);
      else if (word.startsWith('@')) out += span('attr', word);
      else if (WGSL_TYPES.test(word)) out += span('t', word);
      else if (/^\s*\(/.test(src.slice(last))) out += span('f', word);
      else if (/^[A-Z]/.test(word) && keywords === JS_KW) out += span('t', word);
      else out += esc(word);
    } else out += esc(tok);
  }
  return out + esc(src.slice(last));
}

// template literals in this project are mostly WGSL: highlight their contents as WGSL
function template(str) {
  const body = str.slice(1, -1);
  return span('s', '`') + code(body, WGSL_KW) + span('s', '`');
}

function html(src) {
  // inline <script> blocks are highlighted as JavaScript, the rest as markup
  let out = '', last = 0, m;
  const re = /(<script\b[^>]*>)([\s\S]*?)(<\/script>)/g;
  while ((m = re.exec(src))) {
    out += markup(src.slice(last, m.index)) + markup(m[1]);
    out += /type="importmap"/.test(m[1]) ? esc(m[2]) : code(m[2], JS_KW);
    out += markup(m[3]);
    last = re.lastIndex;
  }
  return out + markup(src.slice(last));
}

function markup(src) {
  return esc(src)
    .replace(/(&lt;!--[\s\S]*?--&gt;)/g, '<span class="hl-c">$1</span>')
    .replace(/(&lt;\/?)([\w-]+)([^&]*?)(\/?&gt;)/g, (_, open, name, attrs, close) =>
      `${open}<span class="hl-tag">${name}</span>${attrs.replace(/([\w-]+)(=)("[^"]*")/g, '<span class="hl-attr">$1</span>$2<span class="hl-s">$3</span>')}${close}`);
}

/** Highlight `src` written in `lang` (js, jsx, ts, wgsl, html, sh). Returns HTML. */
export function highlight(src, lang = 'js') {
  if (lang === 'html') return html(src);
  if (lang === 'wgsl') return code(src, WGSL_KW);
  if (lang === 'sh' || lang === 'bash' || lang === 'shell') return esc(src).replace(/(^|\n)(\s*#[^\n]*)/g, '$1<span class="hl-c">$2</span>');
  if (lang === 'json' || lang === 'text' || lang === '') return esc(src);
  return code(src, JS_KW);
}

/** Highlight every <pre data-lang> inside `root` in place. */
export function highlightAll(root = document) {
  for (const pre of root.querySelectorAll('pre[data-lang]')) pre.innerHTML = highlight(pre.textContent, pre.dataset.lang);
}
