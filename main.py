from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

import os


load_dotenv()

def main():
    print("Hello, LangChain!")
    # print(os.getenv("OPENAI_API_KEY"))

    information = """  Jane Austen (/ˈɒstɪn, ˈɔːstɪn/ OST-in, AW-stin; 16 December 1775 – 18 July 1817) was an English writer known primarily for her six novels, which implicitly interpret, critique, and comment on the English landed gentry at the end of the 18th century.

Austen's plots often explore the dependence of women on marriage for the pursuit of favourable social standing and economic security. Her works are implicit critiques of the novels of sensibility of the second half of the 18th century and are part of the transition to 19th-century literary realism.[3][c] Her use of social commentary, realism, wit, and irony have earned her acclaim amongst critics and scholars.

Austen wrote major novels before the age of 22, but she was not published until she was 35. The anonymously published Sense and Sensibility (1811), Pride and Prejudice (1813), Mansfield Park (1814), and Emma (1815) were moderate successes, but they did not bring her public fame in her lifetime. She wrote two other novels—Northanger Abbey and Persuasion, both published posthumously in 1817—and began another, eventually titled Sanditon, but it was left unfinished on her death. She also left behind three volumes of juvenile writings in manuscript, the short epistolary novel Lady Susan, and the unfinished novel The Watsons.

Since her death Austen's novels have rarely been out of print. A significant transition in her reputation occurred in 1833, when they were republished in Richard Bentley's Standard Novels series (illustrated by Ferdinand Pickering and sold as a set). They gradually gained wide acclaim and popular readership. In 1869 her nephew published A Memoir of Jane Austen. Her work has inspired a large number of critical essays and has been included in many literary anthologies.

Her novels have been frequently adapted into film and television. Notable film adaptations include Pride and Prejudice (1940), Sense and Sensibility (1995), Emma (1996), Mansfield Park (1999), Bride and Prejudice (2004), Pride & Prejudice (2005), Love & Friendship (2016), and Emma (2020). Notable television adaptations include Persuasion (1995), Pride and Prejudice (1995), Northanger Abbey (2007), Persuasion (2007), Lost in Austen (2008), Sense and Sensibility (2008), Emma (2009), and Sanditon (2019–2023).
     """

    # And this information here we want to eventually propagate to the LM.
    summary_template = """
    given the information {information} about a person I want you to create :
    1. A short summary
    2. two interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    #llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()