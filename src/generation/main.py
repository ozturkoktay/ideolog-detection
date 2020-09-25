import markovify

# Get raw text as string.
with open("../../data/processed/quotes/part2/muhafazakar_processed.txt") as f:
    text = f.read()

# Build the model.
text_model = markovify.Text(text)

# Print three randomly-generated sentences of no more than 280 characters
for i in range(10):
    print(text_model.make_sentence())
