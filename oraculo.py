import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup

# Funções para carregar documentos
def carrega_site(url):
    if not url:
        return "Erro: Nenhuma URL fornecida."
    
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url  # Adiciona 'https://' se não estiver presente
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Levanta um erro se o status não for 200 (OK)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        texto = soup.get_text(separator=' ', strip=True)
        
        if "Just a moment..." in texto or "Enable JavaScript and cookies to continue" in texto:
            return "Erro: O site exige JavaScript ou cookies para carregar o conteúdo. Por favor, tente novamente ou carregue outro site."
        
        # Trunca o conteúdo do site para evitar exceder o limite de tokens
        max_tokens = 1000  # Ajuste conforme necessário
        texto_truncado = " ".join(texto.split()[:max_tokens])
        return texto_truncado
    except requests.exceptions.RequestException as e:
        return f"Erro ao carregar o site: {e}"

def carrega_youtube(url):
    # Implemente a lógica para carregar vídeos do YouTube
    return "Conteúdo do YouTube carregado."

def carrega_pdf(caminho_arquivo):
    # Implemente a lógica para carregar PDFs
    return "Conteúdo do PDF carregado."

def carrega_csv(caminho_arquivo):
    # Implemente a lógica para carregar CSVs
    return "Conteúdo do CSV carregado."

def carrega_txt(caminho_arquivo):
    # Implemente a lógica para carregar TXTs
    return "Conteúdo do TXT carregado."

TIPOS_ARQUIVOS_VALIDOS = ['Site', 'Youtube', 'Pdf', 'Csv', 'Txt']

CONFIG_MODELOS = {
    'Groq': {
        'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
        'chat': ChatGroq
    },
    'OpenAI': {
        'modelos': ['gpt-4', 'gpt-4-turbo-preview', 'gpt-3.5-turbo'],
        'chat': ChatOpenAI
    }
}

# Inicializa a memória corretamente
MEMORIA = ConversationBufferMemory(return_messages=True)

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        return carrega_site(arquivo)
    elif tipo_arquivo == 'Youtube':
        return carrega_youtube(arquivo)
    elif tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        return carrega_pdf(nome_temp)
    elif tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        return carrega_csv(nome_temp)
    elif tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        return carrega_txt(nome_temp)
    else:
        return "Erro: Tipo de arquivo não suportado."

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    if not api_key:
        st.error("Por favor, insira uma API key válida.")
        return
    
    documento = ""
    if tipo_arquivo in TIPOS_ARQUIVOS_VALIDOS:
        documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    # Verifica se houve erro ao carregar o documento
    if documento.startswith("Erro:"):
        st.error(documento)  # Exibe o erro no Streamlit
        return  # Interrompe a execução da função

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

    print(system_message)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain

def pagina_chat():
    st.header('🤖Bem-vindo ao Oráculo', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carregue o Oráculo')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    
    # Limita o histórico de conversa às últimas 5 mensagens
    historico = memoria.buffer_as_messages
    if len(historico) > 5:
        # Cria uma nova memória com as últimas 5 mensagens
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