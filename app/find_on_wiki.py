import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia("MyProjectName (merlin@example.com)", "en")

def get_information(animal: str) -> tuple:
    page = wiki_wiki.page(animal)

    return page.summary, page.fullurl


if __name__ == "__main__":
    res = get_information("Lion")
    print(res)
