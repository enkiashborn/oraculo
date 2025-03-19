import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import (
    WebBaseLoader, YoutubeLoader, CSVLoader, PyPDFLoader, TextLoader
)

# Definindo os tipos de arquivos válidos
TIPOS_ARQUIVOS_VALIDOS = ['PDF', 'CSV', 'Texto', 'Youtube', 'Site']

# Função para buscar detalhes do vídeo usando a API do YouTube
def busca_detalhes_video(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(
        part='snippet',
        id=video_id
    )
    response = request.execute()
    return response

# Função para buscar legendas pelo YouTube API
def busca_legendas_youtube(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.captions().list(
        part="snippet",
        videoId=video_id
    )
    response = request.execute()
    
    if not response.get("items"):
        return "Erro: Nenhuma legenda disponível para este vídeo."
    
    legenda_id = response["items"][0]["id"]
    request = youtube.captions().download(id=legenda_id)
    response = request.execute()
    
    return response.decode("utf-8")

# Função para carregar modelo de IA
def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, api_key_youtube=None):
    print(f"DEBUG: Provedor={provedor}, Modelo={modelo}, API_KEY={api_key}")
    if not provedor or provedor not in CONFIG_MODELOS:
        st.error("Erro: Provedor inválido ou não selecionado.")
        return

    if not api_key:
        st.error("Por favor, insira uma API key válida.")
        return
    
    documento = carrega_arquivos(tipo_arquivo, arquivo, api_key_youtube)
    
    if documento.startswith("Erro:"):
        st.error(documento)
        return

    system_message = f'''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas de um documento {tipo_arquivo}: 
    ####
    {documento}
    ####
    Utilize as informações fornecidas para basear as suas respostas.'''

    template = ChatPromptTemplate.from_messages([ 
        ('system', system_message),
        ('human', '{input}')
    ])

    try:
        chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
        chain = template | chat
        st.session_state['chain'] = chain
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")

# Sidebar para seleção de modelo e upload de arquivos
def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
            api_key_youtube = st.text_input('Adicione a API key do YouTube')
        else:
            arquivo = st.file_uploader(f'Faça o upload do arquivo {tipo_arquivo.lower()}', type=[f'.{tipo_arquivo.lower()}'])
    
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelos', list(CONFIG_MODELOS.keys()))
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(f'Adicione a API key para o provedor {provedor}',
                                value=st.session_state.get(f'api_key_{provedor}'))
        st.session_state[f'api_key_{provedor}'] = api_key
    
    if st.button('Inicializar Oráculo', use_container_width=True):
        if not provedor:
            st.error("Erro: Nenhum provedor foi selecionado.")
        else:
            carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, api_key_youtube)
    
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = ConversationBufferMemory(return_messages=True)

# Página principal
def main():
    with st.sidebar:
        sidebar()
    st.header('🤖 Bem-vindo ao Oráculo')

if __name__ == '__main__':
    main()
