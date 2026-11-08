// A small Markdown renderer for the docs pages: headings (with GitHub-compatible anchors), fenced
// code (highlighted), tables, nested lists, block quotes, rules, links, bold, italics and inline
// code. Input is our own documentation, but everything is escaped all the same.
import { highlight } from './highlight.js';

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** GitHub's heading anchor: lowercase, punctuation removed, spaces to hyphens. */
export function slug(text) {
  return text.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\p{L}\p{N}\s_-]/gu, '').trim().replace(/\s/g, '-');
}

/**
 * Render markdown to HTML. opts.link(href) may rewrite link targets (e.g. between doc pages).
 * Returns { html, headings: [{ level, text, id }] }.
 */
export function render(md, opts = {}) {
  const link = opts.link || ((h) => h);
  const headings = [];
  const used = new Map();

  const inline = (src) => {
    const codes = [];
    let s = src.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
    s = esc(s);
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, url) => `<img src="${esc(link(url))}" alt="${alt}">`);
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, text, url) => {
      const href = link(url.replace(/&amp;/g, '&'));
      const ext = /^https?:/.test(href);
      return `<a href="${esc(href)}"${ext ? ' rel="noopener"' : ''}>${text}</a>`;
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^\w*])\*([^*\s][^*]*?)\*(?!\w)/g, '$1<em>$2</em>');
    s = s.replace(/(^|[^\w])_([^_\s][^_]*?)_(?!\w)/g, '$1<em>$2</em>');
    return s.replace(/\u0000(\d+)\u0000/g, (_, k) => `<code>${esc(codes[k])}</code>`);
  };

  const lines = md.replace(/\r\n?/g, '\n').split('\n');
  const isBlockStart = (l) => /^(```|#{1,6}\s|>\s?|\s*([-*+]|\d+\.)\s+|\|)/.test(l) || /^(-{3,}|\*{3,})\s*$/.test(l);

  function blocks(ls) {
    let out = '';
    for (let k = 0; k < ls.length;) {
      const l = ls[k];
      if (!l.trim()) { k++; continue; }
      let m;
      if ((m = /^```\s*([\w-]*)/.exec(l))) {
        const lang = m[1] || 'text', body = [];
        for (k++; k < ls.length && !/^```\s*$/.test(ls[k]); k++) body.push(ls[k]);
        k++;
        out += `<pre data-lang="${esc(lang)}">${highlight(body.join('\n'), lang)}</pre>\n`;
      } else if ((m = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(l))) {
        const level = m[1].length, text = m[2];
        let id = slug(text);
        const n = used.get(id) || 0;
        used.set(id, n + 1);
        if (n) id += `-${n}`;
        headings.push({ level, text: text.replace(/`/g, ''), id });
        out += `<h${level} id="${id}"><a class="anchor" href="#${id}" aria-hidden="true">#</a>${inline(text)}</h${level}>\n`;
        k++;
      } else if (/^(-{3,}|\*{3,})\s*$/.test(l)) { out += '<hr>\n'; k++; }
      else if (/^\|/.test(l) && k + 1 < ls.length && /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(ls[k + 1])) {
        const cells = (row) => row.trim().replace(/^\||\|$/g, '').split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, '|'));
        const align = cells(ls[k + 1]).map((c) => (/^:-+:$/.test(c) ? 'center' : /-:$/.test(c) ? 'right' : ''));
        const th = cells(l).map((c, j) => `<th${align[j] ? ` style="text-align:${align[j]}"` : ''}>${inline(c)}</th>`).join('');
        let body = '';
        for (k += 2; k < ls.length && /^\|/.test(ls[k]); k++)
          body += `<tr>${cells(ls[k]).map((c, j) => `<td${align[j] ? ` style="text-align:${align[j]}"` : ''}>${inline(c)}</td>`).join('')}</tr>`;
        out += `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>\n`;
      } else if (/^>\s?/.test(l)) {
        const q = [];
        for (; k < ls.length && /^>\s?/.test(ls[k]); k++) q.push(ls[k].replace(/^>\s?/, ''));
        out += `<blockquote>${blocks(q)}</blockquote>\n`;
      } else if ((m = /^(\s*)([-*+]|\d+\.)\s+/.exec(l))) {
        const indent = m[1].length, ordered = /\d/.test(m[2]);
        const items = [];
        while (k < ls.length) {
          const im = /^(\s*)([-*+]|\d+\.)\s+(.*)$/.exec(ls[k]);
          if (im && im[1].length === indent) { items.push([im[3]]); k++; continue; }
          // a continuation: indented deeper than the marker (nested lists, wrapped text) or a lazy line
          if (items.length && ls[k].trim() && (/^\s/.test(ls[k]) && ls[k].search(/\S/) > indent || (!isBlockStart(ls[k]) && !im))) {
            items[items.length - 1].push(ls[k].slice(Math.min(indent + 2, ls[k].search(/\S/))));
            k++;
            continue;
          }
          break;
        }
        const tag = ordered ? 'ol' : 'ul';
        out += `<${tag}>${items.map((it) => {
          const [first, ...rest] = it;
          const nested = rest.findIndex((r) => /^\s*([-*+]|\d+\.)\s+/.test(r));
          const text = nested < 0 ? [first, ...rest] : [first, ...rest.slice(0, nested)];
          return `<li>${inline(text.join(' '))}${nested < 0 ? '' : blocks(rest.slice(nested))}</li>`;
        }).join('')}</${tag}>\n`;
      } else {
        const p = [];
        for (; k < ls.length && ls[k].trim() && !isBlockStart(ls[k]); k++) p.push(ls[k].trim());
        if (!p.length) { p.push(ls[k].trim()); k++; }
        out += `<p>${inline(p.join(' '))}</p>\n`;
      }
    }
    return out;
  }

  return { html: blocks(lines), headings };
}
