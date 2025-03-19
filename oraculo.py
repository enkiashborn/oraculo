import tempfile
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    WebBaseLoader, YoutubeLoader, CSVLoader, PyPDFLoader, TextLoader
)

# Corrigindo a importação com tratamento de erros
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
except ImportError:
    st.error("Erro: A biblioteca 'youtube_transcript_api' não está instalada. Instale-a com 'pip install youtube-transcript-api'.")
    st.stop()

# Funções para carregar documentos
def carrega_site(url):
    loader = WebBaseLoader(url)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_youtube(video_url):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]  # Extrai o ID do vídeo da URL
        loader = YoutubeLoader(video_id, add_video_info=False, language=['pt', 'en'])
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        return documento
    except TranscriptsDisabled:
        return "Erro: Transcrição desabilitada para este vídeo."
    except Exception as e:
        return f"Erro ao carregar o vídeo do YouTube: {e}"

def carrega_pdf(caminho):
    try:
        loader = PyPDFLoader(caminho)
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        return documento
    except Exception as e:
        return f"Erro ao carregar o PDF: {e}"
    finally:
        if os.path.exists(caminho):
            os.remove(caminho)  # Remove o arquivo temporário após o uso

def carrega_csv(caminho):
    try:
        loader = CSVLoader(caminho)
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        return documento
    except Exception as e:
        return f"Erro ao carregar o CSV: {e}"
    finally:
        if os.path.exists(caminho):
            os.remove(caminho)

def carrega_txt(caminho):
    try:
        loader = TextLoader(caminho)
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        return documento
    except Exception as e:
        return f"Erro ao carregar o TXT: {e}"
    finally:
        if os.path.exists(caminho):
            os.remove(caminho)

TIPOS_ARQUIVOS_VALIDOS = ['Site', 'Youtube', 'Pdf', 'Csv', 'Txt']

CONFIG_MODELOS = {
    'Groq': {
        'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
        'chat': ChatGroq
    },
    'OpenAI': {
        'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
        'chat': ChatOpenAI
    }
}

MEMORIA = ConversationBufferMemory(return_messages=True)

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        return carrega_site(arquivo)
    elif tipo_arquivo == 'Youtube':
        return carrega_youtube(arquivo)
    elif tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            temp_path = temp.name
        return carrega_pdf(temp_path)
    elif tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            temp_path = temp.name
        return carrega_csv(temp_path)
    elif tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            temp_path = temp.name
        return carrega_txt(temp_path)
    else:
        return "Erro: Tipo de arquivo não suportado."

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    if not api_key:
        st.error("Por favor, insira uma API key válida.")
        return
    
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    if documento.startswith("Erro:"):
        st.error(documento)
        return

    system_message = '''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas 
    de um documento {}: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o Oráculo!'''.format(tipo_arquivo, documento)

    st.write("Conteúdo do documento carregado:", documento)  # Log para depuração

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain
    st.success("Oráculo carregado com sucesso! Agora você pode conversar.")

def pagina_chat():
    st.header('🤖Bem-vindo ao Oráculo', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carregue o Oráculo')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    
    historico = memoria.buffer_as_messages
    if len(historico) > 5:
        nova_memoria = ConversationBufferMemory(return_messages=True)
        for mensagem in historico[-5:]:
            if mensagem.type == "human":
                nova_memoria.chat_memory.add_user_message(mensagem.content)
            elif mensagem.type == "ai":
                nova_memoria.chat_memory.add_ai_message(mensagem.content)
        st.session_state['memoria'] = nova_memoria
        memoria = nova_memoria

    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o oráculo')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        try:
            resposta = chat.write_stream(chain.stream({
                'input': input_usuario, 
                'chat_history': memoria.buffer_as_messages
            }))
            
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta)
            st.session_state['memoria'] = memoria
        except Exception as e:
            st.error(f"Erro ao processar a mensagem: {e}")

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
            if arquivo and not verifica_url_youtube(arquivo):
                st.error("Por favor, insira uma URL válida do YouTube.")
        elif tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
        elif tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do arquivo csv', type=['.csv'])
        elif tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo txt', type=['.txt'])
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelos', list(CONFIG_MODELOS.keys()))
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a api key para o provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}'))
        st.session_state[f'api_key_{provedor}'] = api_key
    
    if st.button('Inicializar Oráculo', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = ConversationBufferMemory(return_messages=True)

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()
