
# -*- coding: utf-8 -*-

# Referência Geral para a biblioteca SQLAlchemy:
# A documentação oficial é a fonte primária para todos os conceitos aplicados aqui.
# Link: https://www.sqlalchemy.org/

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- Configuração do Banco de Dados ---

# Define o caminho base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define o caminho para o arquivo do banco de dados SQLite dentro da pasta 'app'
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'app', 'ewaste_log.db')}"
DB_PATH = os.path.join(BASE_DIR, 'app', 'ewaste_log.db')

# --- Criação da Engine e Sessão ---

# A "engine" é o ponto de entrada para o banco de dados.
# Ela gerencia as conexões.
# Referência: https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# A "Session" é a principal interface para persistir e consultar objetos no banco de dados.
# sessionmaker cria uma fábrica de sessões que usaremos para criar sessões individuais.
# Referência: https://docs.sqlalchemy.org/en/20/orm/session_basics.html#session-basics
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# "Base" é uma classe da qual nossos modelos ORM (as tabelas) irão herdar.
# Referência: https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html
Base = declarative_base()

# --- Definição do Modelo (Tabela) ---

class Classification(Base):
    """
    Modelo ORM que representa a tabela 'classifications' no banco de dados.
    Cada atributo da classe corresponde a uma coluna na tabela.
    """
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_name = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)

# --- Funções de Interação com o Banco de Dados ---

def init_db():
    """
    Cria todas as tabelas no banco de dados que herdam de 'Base'.
    Isso é equivalente ao comando "CREATE TABLE IF NOT EXISTS".
    """
    print(f"Inicializando banco de dados em: {DATABASE_URL}")
    # A linha a seguir inspeciona o banco de dados e cria as tabelas que não existem.
    # Referência: https://docs.sqlalchemy.org/en/20/orm/tutorial.html#creating-database-tables
    Base.metadata.create_all(bind=engine)
    print("Banco de dados inicializado com sucesso.")

def log_classification(image_name: str, image_path: str, predicted_class: str):
    """
    Registra o resultado de uma classificação de imagem no banco de dados.
    """
    # Cria uma nova sessão para esta transação específica
    db_session = SessionLocal()
    try:
        # Cria uma instância do nosso modelo Classification
        new_log = Classification(
            image_name=image_name,
            image_path=image_path,
            predicted_class=predicted_class
        )
        # Adiciona o novo registro à sessão
        db_session.add(new_log)
        # Confirma (salva) a transação no banco de dados
        db_session.commit()
        print(f"Classificação para '{image_name}' registrada no banco de dados.")
    except Exception as e:
        print(f"Erro ao registrar classificação: {e}")
        db_session.rollback() # Desfaz a transação em caso de erro
    finally:
        db_session.close() # Sempre fecha a sessão

# --- Bloco de Execução Principal ---

if __name__ == '__main__':
    # Este bloco permite executar o script diretamente para inicializar o DB.
    # É útil para a configuração inicial do projeto.
    print("Executando o módulo de banco de dados diretamente para inicialização...")
    init_db()
