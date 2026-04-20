# 2° Entrega - Tarefa 1: Gerenciador de Pipelines para Ciência de Dados
import os
import pandas as pd
import numpy as np
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, MinMaxScaler, QuantileTransformer, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate
# Flavia: Visualização
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
# Encoding
from sklearn.preprocessing import LabelEncoder
# Normalização
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer
# Seleção de atributos
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import clone # adicionado agora

# Parte - Nathan
class GerenciadorDePipelines:
    def __init__(self, caminho_dataset: str):
        """
        Tarefa 1.1: Setup Base.
        """
        self.caminho_dataset = caminho_dataset
        self.dataset = None
        self.pipelines_gerados = []
        
    def carregar_dados(self):
        try:
            # sep=';' adicionado para lidar com o padrão de CSV da ANP
            self.dataset = pd.read_csv(self.caminho_dataset, sep=';')
            print(f"Dataset carregado com sucesso! Formato: {self.dataset.shape}")
            return self.dataset
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}")
            return None
        

    def _gerar_combinacoes(self, dicionario_tecnicas: dict):
        """
        Tarefa 1.2: Construção do Loop Combinatório.
        """
        chaves = dicionario_tecnicas.keys()
        valores = dicionario_tecnicas.values()
        combinacoes = [dict(zip(chaves, combinacao)) for combinacao in itertools.product(*valores)]
        return combinacoes

    def construir_pipelines(self, dicionario_tecnicas: dict):
        """
        Tarefa 1.3: Integração do Pipeline (Evitando Data Leakage).
        """
        combinacoes = self._gerar_combinacoes(dicionario_tecnicas)
        self.pipelines_gerados = []
        
        for idx, combinacao in enumerate(combinacoes):
            etapas_pipeline = []
            for nome_etapa, tecnica in combinacao.items():
                if tecnica is not None:
                    etapas_pipeline.append((nome_etapa, tecnica))
                else:
                    # Adiciona 'passthrough' para o Baseline conforme requisitos
                    etapas_pipeline.append((nome_etapa, 'passthrough'))
            
            pipeline_atual = Pipeline(etapas_pipeline)
            self.pipelines_gerados.append((f"Combinações_Pipelines -> {idx+1}:", pipeline_atual))
            
        return self.pipelines_gerados
    
# --- BLOCO DE EXECUÇÃO DA TAREFA 1 ---
def aplicar_interpolacao(X):
    """Aplica interpolação apenas em colunas numéricas, ignorando strings."""
    df = pd.DataFrame(X)
    
    # Identifica colunas numéricas
    colunas_numericas = df.select_dtypes(include=['number']).columns.tolist()
    
    # Aplica interpolação apenas nas colunas numéricas
    if colunas_numericas:
        df[colunas_numericas] = df[colunas_numericas].interpolate(
            method='linear', limit_direction='both'
        )
    
    return df.values

# Instanciação (Ajuste o caminho do arquivo aqui) -> caminho para o dataset
gerenciador = GerenciadorDePipelines(caminho_dataset=r'C:\Users\Itallo_Melo\Documents\VSC - Projetos\Projeto - Ciencia de dados\Dataset 2015 - 2025 (Separados)-20260418T194142Z-3-001\Dataset 2015 - 2025 (Separados)\Dataset_Completo_2015_2025.csv')
dados = gerenciador.carregar_dados()

experimentos = {
    'imputacao': [FunctionTransformer(aplicar_interpolacao), None],
    # 'encoding': [
    #     OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
    #     OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    #     None
    # ],
    'normalizacao': [RobustScaler(), MinMaxScaler(), QuantileTransformer(), None],
    'selecao': [SelectKBest(score_func=f_regression, k=5), SelectFromModel(RandomForestRegressor(n_estimators=50)), None],
    'regressor': [KNeighborsRegressor(n_neighbors=7)] 
}

lista_pipelines = gerenciador.construir_pipelines(experimentos)
print(f"Tarefa 1 finalizada: {len(lista_pipelines)} combinações estruturadas.")

# Parte Flávia: Tratamento de nulos e Encoding
# Preparação do Dataframe Base para Modelagem

class PreparadorDeDataframeModelagem:
    """
    Prepara uma cópia limpa do dataframe para o pipeline de pré-processamento,
    extraindo features temporais e realizando amostragem estratificada para
    viabilizar o processamento em memória.
    """

    COLUNAS_RELEVANTES = [
        'Regiao - Sigla',
        'Estado - Sigla',
        'Municipio',
        'Produto',
        'Unidade de Medida',
        'Bandeira',
        'Data da Coleta',
        'Valor de Venda',
        'Valor de Compra',
    ]

    def __init__(self, dataframe: pd.DataFrame, frac_amostra: float = 0.05, seed: int = 42):
        self.dataframe = dataframe
        self.frac_amostra = frac_amostra
        self.seed = seed

    def _selecionar_colunas(self, df: pd.DataFrame) -> pd.DataFrame:
        colunas_existentes = [c for c in self.COLUNAS_RELEVANTES if c in df.columns]
        return df[colunas_existentes].copy()

    def _extrair_features_temporais(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Data da Coleta'] = pd.to_datetime(df['Data da Coleta'], errors='coerce')
        df = df.dropna(subset=['Data da Coleta'])
        df['Ano'] = df['Data da Coleta'].dt.year
        df['Mes'] = df['Data da Coleta'].dt.month
        return df.drop(columns=['Data da Coleta'])

    def _amostrar_estratificado(self, df: pd.DataFrame) -> pd.DataFrame:
        amostras = []
        for _, grupo in df.groupby(['Produto', 'Regiao - Sigla']):
            amostra = grupo.sample(frac=self.frac_amostra, random_state=self.seed)
            amostras.append(amostra)
        return pd.concat(amostras).reset_index(drop=True)

    def preparar(self) -> pd.DataFrame:
        df = self._selecionar_colunas(self.dataframe)
        df = self._extrair_features_temporais(df)
        df = self._amostrar_estratificado(df)
        print(f"✅ Dataframe de modelagem criado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
        return df


# 'dados' é o dataframe carregado pelo gerenciador — era referenciado
# incorretamente como 'dataframe_final' no código original.
preparador = PreparadorDeDataframeModelagem(dados, frac_amostra=0.05)
df_modelo = preparador.preparar()
df_modelo.head(3)
# Imputação — Técnica 1: Remoção de Colunas com Alta Taxa de Nulos

class RemovadorDeColunasPorNulos:
    """
    Técnica 1 de tratamento de nulos: remoção de colunas com percentual
    de valores ausentes acima do limiar definido.
    """

    COLUNAS_PROTEGIDAS = [
        'Regiao - Sigla', 'Estado - Sigla', 'Municipio',
        'Produto', 'Unidade de Medida', 'Bandeira',
        'Valor de Venda', 'Ano', 'Mes',
    ]

    def __init__(self, limiar_percentual: float = 30.0):
        self.limiar_percentual = limiar_percentual
        self.colunas_removidas_: list = []

    def _calcular_percentual_nulos(self, df: pd.DataFrame) -> pd.Series:
        return (df.isnull().sum() / len(df)) * 100

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        percentuais = self._calcular_percentual_nulos(df)
        candidatas = percentuais[percentuais > self.limiar_percentual].index.tolist()
        self.colunas_removidas_ = [c for c in candidatas if c not in self.COLUNAS_PROTEGIDAS]

        df_resultado = df.drop(columns=self.colunas_removidas_)

        print("─" * 60)
        print(" TÉCNICA 1 — REMOÇÃO DE COLUNAS COM EXCESSO DE NULOS")
        print("─" * 60)
        print(f"\n Limiar definido: {self.limiar_percentual}%")

        if self.colunas_removidas_:
            print(f" Colunas removidas ({len(self.colunas_removidas_)}):")
            for col in self.colunas_removidas_:
                print(f"   ✗ '{col}' → {percentuais[col]:.2f}% nulos")
        else:
            print(" Nenhuma coluna não-protegida ultrapassou o limiar.")

        nulos_restantes = df_resultado.isnull().sum().sum()
        print(f"\n Nulos restantes no dataframe: {nulos_restantes:,}")
        print(f" Shape final: {df_resultado.shape[0]:,} × {df_resultado.shape[1]}")
        print(f" Colunas mantidas: {df_resultado.columns.tolist()}")
        return df_resultado


removador = RemovadorDeColunasPorNulos(limiar_percentual=30.0)
df_sem_colunas_nulas = removador.fit_transform(df_modelo)

# Imputação — Técnica 2: Interpolação Linear

class InterpoladorLinear:
    """
    Técnica 2 de tratamento de nulos: interpolação linear para colunas
    numéricas com ausências pontuais, respeitando a ordenação temporal.
    """

    def __init__(self, colunas_temporais: list = None):
        self.colunas_temporais = colunas_temporais or ['Ano', 'Mes']
        self.relatorio_: dict = {}

    def _colunas_numericas_com_nulos(self, df: pd.DataFrame) -> list:
        numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numericas if df[c].isnull().any()]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_resultado = df.copy()

        colunas_ordenacao = [c for c in self.colunas_temporais if c in df_resultado.columns]
        if colunas_ordenacao:
            df_resultado = df_resultado.sort_values(by=colunas_ordenacao).reset_index(drop=True)

        alvos = self._colunas_numericas_com_nulos(df_resultado)

        print("─" * 60)
        print(" TÉCNICA 2 — INTERPOLAÇÃO LINEAR")
        print("─" * 60)

        if not alvos:
            print("\n ✅ Nenhuma coluna numérica com valores ausentes encontrada.")
        else:
            for col in alvos:
                nulos_antes = df_resultado[col].isnull().sum()
                df_resultado[col] = df_resultado[col].interpolate(method='linear', limit_direction='both')
                nulos_depois = df_resultado[col].isnull().sum()
                self.relatorio_[col] = {'antes': nulos_antes, 'depois': nulos_depois}
                print(f"\n  Coluna '{col}':")
                print(f"   → Nulos antes : {nulos_antes:,}")
                print(f"   → Nulos depois: {nulos_depois:,}")

        nulos_totais = df_resultado.isnull().sum().sum()
        print(f"\n Total de nulos restantes: {nulos_totais:,}")
        print(f" Shape final: {df_resultado.shape[0]:,} linhas × {df_resultado.shape[1]} colunas")
        return df_resultado


interpolador = InterpoladorLinear()
df_sem_nulos = interpolador.fit_transform(df_sem_colunas_nulas)

# Encoding — Técnica 1: Label Encoding

class AplicadorDeLabelEncoding:
    """
    Técnica 1 de encoding: Label Encoding para variáveis categóricas
    de alta cardinalidade (Municipio, Bandeira).
    """

    COLUNAS_ALVO = ['Municipio', 'Bandeira', 'Estado - Sigla']

    def __init__(self):
        self.encoders_: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_resultado = df.copy()

        print("─" * 60)
        print(" TÉCNICA 1 DE ENCODING — LABEL ENCODING")
        print("─" * 60)

        for col in self.COLUNAS_ALVO:
            if col not in df_resultado.columns:
                print(f"\n  ⚠ Coluna '{col}' não encontrada — ignorada.")
                continue

            le = LabelEncoder()
            df_resultado[col] = le.fit_transform(df_resultado[col].astype(str))
            self.encoders_[col] = le

            n_classes = len(le.classes_)
            print(f"\n  Coluna '{col}':")
            print(f"   → Categorias únicas: {n_classes:,}")
            print(f"   → Tipo resultante  : {df_resultado[col].dtype}")
            print(f"   → Exemplo          : {le.classes_[:5].tolist()} → {list(range(min(5, n_classes)))}")

        print(f"\n Shape após Label Encoding: {df_resultado.shape[0]:,} linhas × {df_resultado.shape[1]} colunas")
        return df_resultado


label_encoder = AplicadorDeLabelEncoding()
df_label_encoded = label_encoder.fit_transform(df_sem_nulos)

#  Encoding — Técnica 2: Dummy Encoding (One-Hot Encoding)

class AplicadorDeDummyEncoding:
    """
    Técnica 2 de encoding: Dummy Encoding (One-Hot) para variáveis
    categóricas nominais de baixa/média cardinalidade.
    drop_first=True evita multicolinearidade.
    """

    COLUNAS_ALVO = ['Produto', 'Regiao - Sigla', 'Unidade de Medida']

    def __init__(self, drop_first: bool = True):
        self.drop_first = drop_first
        self.colunas_geradas_: list = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_resultado = df.copy()

        print("─" * 60)
        print(" TÉCNICA 2 DE ENCODING — DUMMY ENCODING (ONE-HOT)")
        print("─" * 60)

        colunas_validas = [c for c in self.COLUNAS_ALVO if c in df_resultado.columns]

        shape_antes = df_resultado.shape[1]

        for col in colunas_validas:
            categorias = df_resultado[col].unique()
            print(f"\n  Coluna '{col}':")
            print(f"   → Categorias: {sorted(categorias)}")

        df_resultado = pd.get_dummies(
            df_resultado,
            columns=colunas_validas,
            drop_first=self.drop_first,
            dtype=int
        )

        novas_colunas = df_resultado.shape[1] - shape_antes + len(colunas_validas)
        print(f"\n  Colunas dummies criadas: {df_resultado.shape[1] - shape_antes + len(colunas_validas)}")
        print(f" Shape final: {df_resultado.shape[0]:,} linhas × {df_resultado.shape[1]} colunas")

        self.colunas_geradas_ = df_resultado.columns.tolist()
        return df_resultado


dummy_encoder = AplicadorDeDummyEncoding(drop_first=True)
df_encoded = dummy_encoder.fit_transform(df_label_encoded)

print("\n Colunas após encoding completo:")
print(df_encoded.columns.tolist())

# Escalonamento (Normalização)
# Preparação — Separação de Features e Alvo

# Definimos a variável alvo e as features para escalonamento
COLUNA_ALVO = 'Valor de Venda'

FEATURES_NUMERICAS = ['Ano', 'Mes']

# Verificamos quais colunas numéricas estão disponíveis
print("Colunas disponíveis no dataframe codificado:")
print(df_encoded.dtypes.to_string())

# Normalização — Técnica 1: RobustScaler
class AplicadorRobustScaler:
    """
    Técnica 1 de normalização: RobustScaler.
    Usa mediana e IQR — ideal para dados com outliers expressivos
    como flutuações de preço por crises econômicas.
    """

    def __init__(self, colunas: list):
        self.colunas = colunas
        self.scaler = RobustScaler()
        self.df_escalado_: pd.DataFrame = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_resultado = df.copy()

        colunas_validas = [c for c in self.colunas if c in df_resultado.columns]
        valores_escalonados = self.scaler.fit_transform(df_resultado[colunas_validas])
        df_resultado[colunas_validas] = valores_escalonados

        self.df_escalado_ = df_resultado

        print("─" * 60)
        print(" TÉCNICA 1 — ROBUST SCALER")
        print("─" * 60)
        print(f"\n Colunas escalonadas: {colunas_validas}")
        for i, col in enumerate(colunas_validas):
            print(f"   '{col}': centro={self.scaler.center_[i]:.4f} | escala={self.scaler.scale_[i]:.4f}")
        print(f"\n Estatísticas pós-escalonamento:")
        display(df_resultado[colunas_validas].describe().round(4))
        return df_resultado

    def plotar_comparacao(self, df_original: pd.DataFrame) -> None:
        colunas_validas = [c for c in self.colunas if c in df_original.columns]
        if not colunas_validas:
            return

        col = colunas_validas[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df_original[col].dropna(), ax=axes[0], kde=True, color='#457B9D')
        axes[0].set_title(f"'{col}' — Antes (original)", fontweight='bold')
        axes[0].set_xlabel('Valor')

        sns.histplot(self.df_escalado_[col].dropna(), ax=axes[1], kde=True, color='#E63946')
        axes[1].set_title(f"'{col}' — Depois (RobustScaler)", fontweight='bold')
        axes[1].set_xlabel('Valor escalado')

        plt.suptitle("RobustScaler: Comparação da Distribuição", fontsize=13, fontweight='bold')
        sns.despine()
        plt.tight_layout()
        plt.show()


COLS_PARA_ESCALAR = [COLUNA_ALVO] + FEATURES_NUMERICAS

# Converte a coluna alvo para numérico antes de aplicar o scaler.
# Remove espaços e formata números com vírgula decimal.
if COLUNA_ALVO in df_encoded.columns:
    df_encoded[COLUNA_ALVO] = pd.to_numeric(
        df_encoded[COLUNA_ALVO]
            .astype(str)
            .str.strip()
            .str.replace(r'\.(?=\d{3}(?:\D|$))', '', regex=True)
            .str.replace(',', '.', regex=False),
        errors='coerce'
    )
    n_coercidos = df_encoded[COLUNA_ALVO].isnull().sum()
    print(f"Coluna '{COLUNA_ALVO}' convertida para numérico; valores não convertidos: {n_coercidos}")

robust = AplicadorRobustScaler(colunas=COLS_PARA_ESCALAR)
df_robust = robust.fit_transform(df_encoded)
robust.plotar_comparacao(df_encoded)

# Normalização — Técnica 2: MinMaxScaler

class AplicadorMinMaxScaler:
    """
    Técnica 2 de normalização: MinMaxScaler.
    Escala os dados para [0, 1] — indicado para KNN e redes neurais.
    """
    def __init__(self, colunas: list, feature_range: tuple = (0, 1)):
        self.colunas = colunas
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.df_escalado_: pd.DataFrame = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_resultado = df.copy()

        colunas_validas = [c for c in self.colunas if c in df_resultado.columns]
        valores_escalonados = self.scaler.fit_transform(df_resultado[colunas_validas])
        df_resultado[colunas_validas] = valores_escalonados
        self.df_escalado_ = df_resultado

        print("─" * 60)
        print(" TÉCNICA 2 — MIN-MAX SCALER")
        print("─" * 60)
        print(f"\n Colunas escalonadas: {colunas_validas}")
        print(f"\n Intervalo de saída : {self.scaler.feature_range}")
        for i, col in enumerate(colunas_validas):
            print(f"   '{col}': min_orig={self.scaler.data_min_[i]:.4f} | max_orig={self.scaler.data_max_[i]:.4f}")
        print(f"\n Estatísticas pós-escalonamento:")
        display(df_resultado[colunas_validas].describe().round(4))
        return df_resultado

    def plotar_comparacao(self, df_original: pd.DataFrame) -> None:
        colunas_validas = [c for c in self.colunas if c in df_original.columns]
        if not colunas_validas:
            return

        col = colunas_validas[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df_original[col].dropna(), ax=axes[0], kde=True, color='#2A9D8F')
        axes[0].set_title(f"'{col}' — Antes (original)", fontweight='bold')
        axes[0].set_xlabel('Valor')

        sns.histplot(self.df_escalado_[col].dropna(), ax=axes[1], kde=True, color='#F4A261')
        axes[1].set_title(f"'{col}' — Depois (MinMaxScaler)", fontweight='bold')
        axes[1].set_xlabel('Valor escalado [0, 1]')

        plt.suptitle("MinMaxScaler: Comparação da Distribuição", fontsize=13, fontweight='bold')
        sns.despine()
        plt.tight_layout()
        plt.show()


minmax = AplicadorMinMaxScaler(colunas=COLS_PARA_ESCALAR)
df_minmax = minmax.fit_transform(df_encoded)
minmax.plotar_comparacao(df_encoded)

# Normalização — Técnica 3: QuantileTransformer
class AplicadorQuantileTransformer:
    """
    Técnica 3 de normalização: QuantileTransformer.
    Transforma a distribuição para uniforme — robusto a outliers expressivos.
    """
    def __init__(self, colunas: list, output_distribution: str = 'uniform', n_quantiles: int = 1000):
        self.colunas = colunas
        self.scaler = QuantileTransformer(
            output_distribution=output_distribution,
            n_quantiles=n_quantiles,
            random_state=42
        )
        self.df_escalado_: pd.DataFrame = None
        self.output_distribution = output_distribution

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_resultado = df.copy()

        colunas_validas = [c for c in self.colunas if c in df_resultado.columns]
        valores_escalonados = self.scaler.fit_transform(df_resultado[colunas_validas])
        df_resultado[colunas_validas] = valores_escalonados
        self.df_escalado_ = df_resultado

        print("─" * 60)
        print(" TÉCNICA 3 — QUANTILE TRANSFORMER")
        print("─" * 60)
        print(f"\n Colunas escalonadas     : {colunas_validas}")
        print(f" Distribuição de saída   : {self.output_distribution}")
        print(f" Número de quantis       : {self.scaler.n_quantiles_}")
        print(f"\n Estatísticas pós-escalonamento:")
        display(df_resultado[colunas_validas].describe().round(4))
        return df_resultado

    def plotar_comparacao(self, df_original: pd.DataFrame) -> None:
        colunas_validas = [c for c in self.colunas if c in df_original.columns]
        if not colunas_validas:
            return

        col = colunas_validas[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df_original[col].dropna(), ax=axes[0], kde=True, color='#9B5DE5')
        axes[0].set_title(f"'{col}' — Antes (original)", fontweight='bold')
        axes[0].set_xlabel('Valor')

        sns.histplot(self.df_escalado_[col].dropna(), ax=axes[1], kde=True, color='#F72585')
        axes[1].set_title(f"'{col}' — Depois (QuantileTransformer)", fontweight='bold')
        axes[1].set_xlabel(f'Valor ({self.output_distribution})')

        plt.suptitle("QuantileTransformer: Comparação da Distribuição", fontsize=13, fontweight='bold')
        sns.despine()
        plt.tight_layout()
        plt.show()


quantile = AplicadorQuantileTransformer(colunas=COLS_PARA_ESCALAR, output_distribution='uniform')
df_quantile = quantile.fit_transform(df_encoded)
quantile.plotar_comparacao(df_encoded) 

# Comparativo Visual das Três Técnicas de Normalização
class ComparadorDeEscalonamentos:
    """
    Visualiza lado a lado o efeito das três técnicas de normalização
    sobre a variável alvo.
    """

    def __init__(self, coluna_alvo: str = 'Valor de Venda'):
        self.coluna_alvo = coluna_alvo

    def plotar(
        self,
        df_original: pd.DataFrame,
        df_robust: pd.DataFrame,
        df_minmax: pd.DataFrame,
        df_quantile: pd.DataFrame
    ) -> None:
        frames = {
            'Original': df_original,
            'RobustScaler': df_robust,
            'MinMaxScaler': df_minmax,
            'QuantileTransformer': df_quantile,
        }

        cores = ['#457B9D', '#E63946', '#F4A261', '#F72585']

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(
            f"Distribuição de '{self.coluna_alvo}': Comparativo de Normalização",
            fontsize=14, fontweight='bold', y=1.02
        )

        for ax, (titulo, df_), cor in zip(axes, frames.items(), cores):
            if self.coluna_alvo not in df_.columns:
                ax.set_visible(False)
                continue
            sns.histplot(df_[self.coluna_alvo].dropna(), ax=ax, kde=True, color=cor, bins=40)
            ax.set_title(titulo, fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frequência')
            sns.despine(ax=ax)

        plt.tight_layout()
        plt.show()


comparador = ComparadorDeEscalonamentos(coluna_alvo=COLUNA_ALVO)
comparador.plotar(df_encoded, df_robust, df_minmax, df_quantile)

# Seleção e Redução de Atributos
# Preparação — Separação X / y para Seleção de Features

# df_robust já foi gerado acima pelo AplicadorRobustScaler —
# usamos ele diretamente como base para a etapa de seleção de features.
df_para_selecao = df_robust.copy()

# Separamos features (X) e alvo (y)
X = df_para_selecao.drop(columns=[COLUNA_ALVO])
y = df_para_selecao[COLUNA_ALVO]

# Garantimos apenas colunas numéricas
X = X.select_dtypes(include=[np.number])

print(f"Features disponíveis ({X.shape[1]} colunas):")
print(X.columns.tolist())
print(f"\n Alvo: '{COLUNA_ALVO}' | Shape X: {X.shape} | Shape y: {y.shape}")

# Redução — Técnica 1: Matriz de Correlação + SelectKBest
class SeletorPorCorrelacaoESelectKBest:
    """
    Técnica 1 de seleção de atributos:
    1. Remove features com correlação mútua > limiar (multicolinearidade).
    2. Seleciona as K melhores features via SelectKBest (f_regression).
    """
    def __init__(self, limiar_correlacao: float = 0.90, k_melhores: int = 10):
        self.limiar_correlacao = limiar_correlacao
        self.k_melhores = k_melhores
        self.colunas_removidas_por_correlacao_: list = []
        self.features_selecionadas_: list = []
        self.selector_: SelectKBest = None

    def _remover_por_multicolinearidade(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove uma das features de cada par altamente correlacionado."""
        matriz_corr = X.corr().abs()
        triangulo_superior = matriz_corr.where(
            np.triu(np.ones(matriz_corr.shape), k=1).astype(bool)
        )
        para_remover = [
            col for col in triangulo_superior.columns
            if any(triangulo_superior[col] > self.limiar_correlacao)
        ]
        self.colunas_removidas_por_correlacao_ = para_remover
        return X.drop(columns=para_remover)

    def _plotar_matriz_correlacao(self, X: pd.DataFrame) -> None:
        corr = X.corr()
        mascara = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(max(8, len(X.columns)), max(6, len(X.columns) - 2)))
        sns.heatmap(
            corr, mask=mascara, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5,
            annot_kws={'size': 9}
        )
        plt.title('Matriz de Correlação (features após remoção de multicolinearidade)',
                  fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()

    def _aplicar_select_k_best(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        k = min(self.k_melhores, X.shape[1])
        self.selector_ = SelectKBest(score_func=f_regression, k=k)
        self.selector_.fit(X, y)

        scores = pd.DataFrame({
            'Feature': X.columns,
            'F-Score': self.selector_.scores_,
            'p-valor': self.selector_.pvalues_
        }).sort_values('F-Score', ascending=False)

        self.features_selecionadas_ = X.columns[self.selector_.get_support()].tolist()
        return scores

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        print("─" * 60)
        print(" TÉCNICA 1 — CORRELAÇÃO + SELECTKBEST")
        print("─" * 60)

        # Passo 1: Remoção por multicolinearidade
        X_sem_redundantes = self._remover_por_multicolinearidade(X)
        print(f"\n [Passo 1] Remoção por multicolinearidade (limiar |r| > {self.limiar_correlacao}):")
        if self.colunas_removidas_por_correlacao_:
            print(f"   Removidas: {self.colunas_removidas_por_correlacao_}")
        else:
            print("   Nenhuma feature removida.")
        print(f"   Features restantes: {X_sem_redundantes.shape[1]}")

        # Visualização
        self._plotar_matriz_correlacao(X_sem_redundantes)

        # Passo 2: SelectKBest
        print(f"\n [Passo 2] SelectKBest — Top {self.k_melhores} features (f_regression):")
        scores = self._aplicar_select_k_best(X_sem_redundantes, y)
        display(scores.round(4))

        # Plot F-Scores
        plt.figure(figsize=(10, 5))
        cores_barras = ['#E63946' if f in self.features_selecionadas_ else '#ADB5BD'
                        for f in scores['Feature']]
        ax = sns.barplot(data=scores, x='F-Score', y='Feature',
                         palette=cores_barras, hue='Feature', legend=False)
        ax.set_title(f'F-Score por Feature (vermelho = selecionada no Top {self.k_melhores})',
                     fontweight='bold')
        sns.despine()
        plt.tight_layout()
        plt.show()

        print(f"\n Features selecionadas ({len(self.features_selecionadas_)}):")
        print(f"   {self.features_selecionadas_}")

        X_final = X_sem_redundantes[self.features_selecionadas_]
        print(f"\n Shape final de X: {X_final.shape}")
        return X_final


seletor_correlacao = SeletorPorCorrelacaoESelectKBest(limiar_correlacao=0.90, k_melhores=10)
X_selecionado_t1 = seletor_correlacao.fit_transform(X, y)

# Redução — Feature Importance (Random Forest)
class SeletorPorFeatureImportance:
    """
    Técnica 2 de seleção de atributos: Feature Importance com Random Forest.
    Captura relações não-lineares e seleciona features acima de um limiar
    mínimo de importância relativa.
    """
    def __init__(self, limiar_importancia: float = 0.001, n_estimators: int = 100, seed: int = 42):
        self.limiar_importancia = limiar_importancia
        self.n_estimators = n_estimators
        self.seed = seed
        self.modelo_: RandomForestRegressor = None
        self.importancias_: pd.DataFrame = None
        self.features_selecionadas_: list = []

    def _treinar_modelo(self, X: pd.DataFrame, y: pd.Series) -> None:
        print(" Treinando Random Forest... (aguarde)")
        self.modelo_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=8,
            random_state=self.seed,
            n_jobs=-1
        )
        self.modelo_.fit(X, y)

    def _calcular_importancias(self, X: pd.DataFrame) -> pd.DataFrame:
        importancias = pd.DataFrame({
            'Feature': X.columns,
            'Importância': self.modelo_.feature_importances_
        }).sort_values('Importância', ascending=False).reset_index(drop=True)
        importancias['Importância (%)'] = (importancias['Importância'] * 100).round(2)
        return importancias

    def _plotar_importancias(self) -> None:
        df_plot = self.importancias_.copy()
        df_plot['Selecionada'] = df_plot['Feature'].isin(self.features_selecionadas_)

        plt.figure(figsize=(12, max(5, len(df_plot) * 0.4 + 1)))
        cores_barras = ['#2A9D8F' if s else '#ADB5BD' for s in df_plot['Selecionada']]
        ax = sns.barplot(data=df_plot, x='Importância (%)', y='Feature',
                         palette=cores_barras, hue='Feature', legend=False)

        ax.axvline(
            x=self.limiar_importancia * 100,
            color='#E63946', linestyle='--', linewidth=1.5,
            label=f'Limiar = {self.limiar_importancia*100:.1f}%'
        )
        ax.legend(fontsize=10)
        ax.set_title(
            f'Feature Importance — Random Forest\n(verde = selecionada | limiar = {self.limiar_importancia*100:.1f}%)',
            fontweight='bold'
        )
        sns.despine()
        plt.tight_layout()
        plt.show()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        print("─" * 60)
        print(" TÉCNICA 2 — FEATURE IMPORTANCE (RANDOM FOREST)")
        print("─" * 60)

        self._treinar_modelo(X, y)
        self.importancias_ = self._calcular_importancias(X)
        self.features_selecionadas_ = (
            self.importancias_[self.importancias_['Importância'] >= self.limiar_importancia]['Feature']
            .tolist()
        )

        print(f"\n Limiar mínimo de importância: {self.limiar_importancia*100:.1f}%")
        print(f" Árvores treinadas          : {self.n_estimators}")
        print(f"\n Ranking completo de importância:")
        display(self.importancias_)

        self._plotar_importancias()

        print(f"\n Features selecionadas ({len(self.features_selecionadas_)}):")
        print(f"   {self.features_selecionadas_}")

        X_final = X[self.features_selecionadas_]
        print(f"\n Shape final de X: {X_final.shape}")
        return X_final


seletor_rf = SeletorPorFeatureImportance(
    limiar_importancia=0.001,
    n_estimators=100
)
X_selecionado_t2 = seletor_rf.fit_transform(X, y)

# Comparativo entre as Duas Técnicas de Seleção
class ComparadorDeTecnicasDeSelecao:
    """
    Compara os resultados das duas técnicas de seleção de atributos,
    destacando concordâncias e divergências.
    """
    def __init__(
        self,
        seletor_correlacao: SeletorPorCorrelacaoESelectKBest,
        seletor_rf: SeletorPorFeatureImportance
    ):
        self.t1_features = set(seletor_correlacao.features_selecionadas_)
        self.t2_features = set(seletor_rf.features_selecionadas_)

    def gerar_relatorio(self) -> None:
        concordancia   = self.t1_features & self.t2_features
        apenas_t1      = self.t1_features - self.t2_features
        apenas_t2      = self.t2_features - self.t1_features
        uniao          = self.t1_features | self.t2_features

        print("─" * 60)
        print(" COMPARATIVO ENTRE AS TÉCNICAS DE SELEÇÃO")
        print("─" * 60)

        print(f"\n [T1] Correlação + SelectKBest    : {len(self.t1_features)} features")
        print(f" [T2] Feature Importance (RF)     : {len(self.t2_features)} features")
        print(f"\n ✅ Concordância (em ambas)        : {sorted(concordancia)}")
        print(f" ℹ  Apenas em T1                  : {sorted(apenas_t1)}")
        print(f" ℹ  Apenas em T2                  : {sorted(apenas_t2)}")
        print(f"\n União (recomendação conservadora) : {sorted(uniao)}")
        print(f" Interseção (recomendação restrita): {sorted(concordancia)}")

        # Diagrama de Venn simplificado via barras
        df_comp = pd.DataFrame({
            'Feature': sorted(uniao),
            'T1 — Correlação + SelectKBest': [1 if f in self.t1_features else 0 for f in sorted(uniao)],
            'T2 — Feature Importance (RF)' : [1 if f in self.t2_features else 0 for f in sorted(uniao)],
        }).set_index('Feature')

        df_comp['Ambas'] = (df_comp.sum(axis=1) == 2).astype(int)

        plt.figure(figsize=(12, max(5, len(df_comp) * 0.45 + 1)))
        cmap_custom = {0: '#ADB5BD', 1: '#457B9D'}

        ax = df_comp[['T1 — Correlação + SelectKBest', 'T2 — Feature Importance (RF)']].plot(
            kind='barh', figsize=(12, max(5, len(df_comp) * 0.45 + 1)),
            color=['#457B9D', '#2A9D8F'], edgecolor='white', alpha=0.85
        )
        ax.set_title('Features Selecionadas por Técnica\n(1 = selecionada | 0 = não selecionada)',
                     fontweight='bold')
        ax.set_xlabel('Selecionada')
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.8)
        sns.despine()
        plt.tight_layout()
        plt.show()

        # Recomendação final
        print("\n" + "═" * 60)
        print(" RECOMENDAÇÃO FINAL DE FEATURES")
        print("═" * 60)
        features_recomendadas = sorted(concordancia) if concordancia else sorted(uniao)
        print(f"\n Critério: interseção das duas técnicas.")
        print(f" Features recomendadas ({len(features_recomendadas)}):")
        for f in features_recomendadas:
            print(f"   → {f}")


comparador_selecao = ComparadorDeTecnicasDeSelecao(seletor_correlacao, seletor_rf)
comparador_selecao.gerar_relatorio()

# Parte Bruno
# Separação dos Dados
target = "Valor de Venda"

# O dataframe base para o KNN é df_encoded — já passou por todo o
# pré-processamento da Flávia (nulos, encoding) mas ainda sem escalonamento
# avulso, pois cada pipeline do Nathan já carrega seu próprio scaler internamente.
X = df_encoded.drop(columns=[target]).select_dtypes(include=[np.number])
y = df_encoded[target]

# Reduzir amostra para acelerar os experimentos com KNN
# KNN é computacionalmente intensivo O(n*k) por predição
amostra_size = 5000  # Usar 5000 amostras para os experimentos
np.random.seed(42)
indices = np.random.choice(len(X), size=min(amostra_size, len(X)), replace=False)
X = X.iloc[indices].reset_index(drop=True)
y = y.iloc[indices].reset_index(drop=True)

# Classe do Modelo
class ModeloKNN:
    def __init__(self, k=7):
        self.k = k

    def criar_modelo(self):
        return KNeighborsRegressor(n_neighbors=self.k)
    

# Classe de Avaliação
class AvaliadorDeModelos:
    def __init__(self, cv=5):
        self.cv = cv

    def avaliar(self, pipeline, X, y):
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=self.cv,
            scoring=["neg_mean_squared_error", "r2"],
            return_train_score=False
        )

        resultados = {}

        for chave in scores:
            if "test" in chave:
                resultados[chave] = {
                    "media": scores[chave].mean(),
                    "std": scores[chave].std()
                }

        return resultados
    
#  Classe de Execução
class ExecutorDeExperimentos:
    def __init__(self, avaliador):
        self.avaliador = avaliador
        self.resultados = []

    def executar(self, pipelines, X, y):
        for nome, pipeline in pipelines.items():
            resultado = self.avaliador.avaliar(pipeline, X, y)

            registro = {"pipeline": nome}

            for metrica, valores in resultado.items():
                registro[f"{metrica}_media"] = valores["media"]
                registro[f"{metrica}_std"] = valores["std"]

            self.resultados.append(registro)

        return pd.DataFrame(self.resultados)
    
# Integração dos Pipelines
# O dicionário 'experimentos' já inclui o regressor KNN na chave 'regressor',
# então os pipelines em 'lista_pipelines' já vêm com o modelo incluído.
# Apenas convertemos a lista de tuplas para dicionário.
pipelines = {nome: pipe for nome, pipe in lista_pipelines}

# Execução dos Experimentos
avaliador = AvaliadorDeModelos(cv=5)
executor = ExecutorDeExperimentos(avaliador)

resultados_df = executor.executar(pipelines, X, y)

# Exibir resultados
print("\n" + "═" * 60)
print(" RESULTADOS DOS EXPERIMENTOS COM KNN")
print("═" * 60)
print(resultados_df.to_string())
