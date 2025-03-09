import anthropic

client = anthropic.Anthropic(
    # 默认为 os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)
