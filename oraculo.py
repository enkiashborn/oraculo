import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.document_loaders import (
    WebBaseLoader, YoutubeLoader, CSVLoader, PyPDFLoader, TextLoader
)

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
    
    # Pegando o ID da legenda
    legenda_id = response["items"][0]["id"]

    # Baixando a legenda
    request = youtube.captions().download(id=legenda_id)
    response = request.execute()
    
    return response.decode("utf-8")  # Retorna a legenda como texto

# Função para carregar vídeo do YouTube com API
def carrega_youtube(video_url, api_key):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]  # Extrai o ID do vídeo da URL

        # Busca detalhes do vídeo
        detalhes_video = busca_detalhes_video(api_key, video_id)
        if not detalhes_video['items']:
            return "Erro: Vídeo não encontrado."

        titulo = detalhes_video['items'][0]['snippet']['title']
        descricao = detalhes_video['items'][0]['snippet']['description']
        
        # Busca legendas via API do YouTube
        transcricao = busca_legendas_youtube(api_key, video_id)

        return f"📺 **Título do Vídeo:** {titulo}\n\n📝 **Descrição:** {descricao}\n\n🗣 **Legendas:**\n{transcricao}"
    
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

def carrega_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_txt(caminho):
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

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

# Inicializa a memória corretamente
MEMORIA = ConversationBufferMemory(return_messages=True)

def carrega_arquivos(tipo_arquivo, arquivo, api_key_youtube=None):
    if tipo_arquivo == 'Site':
        return carrega_site(arquivo)
    elif tipo_arquivo == 'Youtube':
        if not api_key_youtube:
            return "Erro: Chave da API do YouTube não fornecida."
        return carrega_youtube(arquivo, api_key_youtube)
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

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, api_key_youtube=None):
    if not api_key:
        st.error("Por favor, insira uma API key válida.")
        return
    
    if provedor not in CONFIG_MODELOS:
        st.error(f"Erro: O provedor '{provedor}' não é válido.")
        return
    
    documento = carrega_arquivos(tipo_arquivo, arquivo, api_key_youtube)
    
    if documento.startswith("Erro:"):
        st.error(documento)
        return

    system_message = f'''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas 
    de um documento {tipo_arquivo}: 

    ####
    {documento}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver "$" na sua saída, substitua por "S".

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o Oráculo!'''

    print("Conteúdo do system_message:", system_message)  # Log para depuração

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

])

# Criando o modelo
chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)

# Corrigindo a pipeline para passar apenas "input" (sem "chat_history")
chain = template | chat
st.session_state['chain'] = chain

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
            print("Variáveis passadas ao chain.stream:", {  # Log para depuração
                'input': input_usuario,
                'chat_history': memoria.buffer_as_messages
            })
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
            api_key_youtube = st.text_input('Adicione a API key do YouTube')
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
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, api_key_youtube)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = ConversationBufferMemory(return_messages=True)

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()
