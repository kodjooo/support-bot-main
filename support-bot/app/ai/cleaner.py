import re


def clean_response(text: str) -> str:
    """Убирает Markdown-форматирование и citation-ссылки OpenAI."""
    pattern = (
        r'\*\*(.*?)\*\*'        # **жирный**
        r'|__(.*?)__'            # __курсив__
        r'|~~(.*?)~~'           # ~~зачёркнутый~~
        r'|\[.*?†source\]'      # [text†source]
        r'|【\d+:\d+†source】'  # 【1:2†source】
        r'|【\d+†source】'      # 【1†source】
    )

    def replacer(m: re.Match) -> str:
        for i in range(1, 4):
            if m.group(i) is not None:
                return m.group(i)
        return ""

    return re.sub(pattern, replacer, text)
