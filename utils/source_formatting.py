def format_source(source: dict, max_chars: int = 800, max_lines: int = 40) -> dict:
    """Javobda ko'rsatiladigan manba (mahsulot tavsifi) matnini qisqartiradi."""
    text = source.get('text', '')
    metadata = source.get('metadata') or {}
    if text:
        text = _trim_source_text(text, max_chars=max_chars, max_lines=max_lines)
        source = dict(source)
        source['text'] = text
        # Fayl nomidan mahsulot nomini chiqarib beramiz (masalan: ceramide_cream.md -> ceramide cream)
        path = metadata.get('path')
        if path and 'product' not in metadata:
            name = path.rsplit('/', 1)[-1].rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ')
            source['metadata'] = {**metadata, 'product': name}
    return source


def _trim_source_text(text: str, max_chars: int, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ['...']
    trimmed = '\n'.join(lines)
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rstrip() + '\n...'
    return trimmed
