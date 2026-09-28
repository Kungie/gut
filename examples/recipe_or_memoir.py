"""Where is the recipe? Finds the pages that make you scroll through a life story first.

    python examples/recipe_or_memoir.py                    # NLI model first, then Qwen3-0.6B
    python examples/recipe_or_memoir.py --backend jev      # NLI model first, then Jev

A `gut.Cascade` in miniature. The 70M-parameter NLI model reads every page and is certain about
the easy ones -- there are ingredients or there are not. The pages it cannot call go on to a
language model. A strict band keeps the cheap model honest: only a near-certain answer stays.
"""

from __future__ import annotations

import gut

PAGES = [
    "Ingredients: 200 g flour, 2 eggs, 300 ml milk, pinch of salt. Whisk, rest 20 minutes, fry.",
    "Every autumn my grandmother's kitchen smelled of apples. She grew up on a farm in Normandy, "
    "where her father kept bees and her mother sang while shelling peas. Years later, in a tiny "
    "Paris flat, she taught me that patience is an ingredient too. That lesson, and this tart, "
    "stayed with me through university, two moves and one very bad haircut. Ingredients: 4 apples, "
    "1 sheet of pastry, 50 g butter, 3 tbsp sugar.",
    "This one-pot pasta is our weeknight hero. Ingredients: 250 g spaghetti, 1 can tomatoes, "
    "2 cloves garlic, basil. Everything into the pot, 12 minutes, done.",
    "Let me tell you about the summer of 2009.",
]


def verdicts(pages: list[str]) -> list[str]:
    recipe = gut.each(pages).likely("lists ingredients")
    story = gut.each(pages).likely("tells a personal story")
    lines = []
    for has_recipe, has_story in zip(recipe, story, strict=True):
        if has_recipe and has_story:
            lines.append("Recipe found, after a life story. Scroll on.")
        elif has_recipe:
            lines.append("Straight to the recipe. A rare and beautiful thing.")
        elif has_story:
            lines.append("No recipe. Only a memoir.")
        else:
            lines.append("Neither a recipe nor a story. What is this page?")
    return lines


def main() -> None:
    from _pick import BACKENDS, backend_from_argv

    bigger = backend_from_argv(__doc__ or "", default="qwen")
    cascade = gut.Cascade(BACKENDS["nli"][1](), bigger, unsure_band=(0.05, 0.95))
    gut.configure(backend=cascade)
    for page, line in zip(PAGES, verdicts(PAGES), strict=True):
        print(f"{page[:52]:<52}...  {line}")
    print("\nanswers per model:", dict(cascade.answered_by))


if __name__ == "__main__":
    main()
