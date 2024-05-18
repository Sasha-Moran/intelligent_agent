import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia("MyProjectName (merlin@example.com)", "en")

def get_information(animal):
    page = wiki_wiki.page(animal)

    return page.summary


if __name__ == "__main__":
    res = get_information("Lion")
    print(res)
