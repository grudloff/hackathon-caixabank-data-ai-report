from setuptools import setup, find_packages

setup(
    name="hackathon-caixabank-data-ai-report",
    version="0.1.0",
    description="A project for the Caixabank Data AI Hackathon",
    author="Gabriel Rudloff",
    author_email="gabriel.rudloff@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.3",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "langchain==0.3.3",
        "langchain-ollama==0.2.0",
        "ollama==0.3.3",
        "requests==2.32.3",
        "fpdf2==2.8.1",
        "pytest==8.3.3",
        "tqdm",
    ],
    python_requires="==3.10.12",
)