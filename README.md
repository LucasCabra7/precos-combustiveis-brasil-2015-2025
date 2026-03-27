<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=66CDAA&height=120&section=header"/>

# Análise Exploratória e Modelagem Preditiva dos Preços de Combustíveis no Brasil (2015–2025)

Projeto desenvolvido para a disciplina de **Algoritmos**, utilizando a linguagem **C** e os princípios fundamentais da **Programação Orientada a Objetos e Algoritmos**.

---

## 📌  Descrição do Dataset

O Projeto tem como objetivo analisar e compreender o comportamento dos preços de combustíveis no Brasil ao longo do tempo, utilizando dados públicos disponibilizados pela [Agência Nacional do Petróleo, Gás Natural e Biocombustíveis](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis).

O conjunto de dados utilizado consiste na série histórica de preços de combustíveis, com informações coletadas semanalmente em diversos postos distribuídos por estados e municípios brasileiros. A base contempla variáveis relevantes, como localização geográfica (estado e município), região, tipo de combustível, datas de coleta e valores de compra e venda.

As **regiões brasileiras** consideradas na base são:

* Norte (N)
* Nordeste (NE)
* Sudeste (SE)
* Sul (S)
* Centro-Oeste (CO)

Já os **produtos analisados** incluem:

* Gasolina Comum
* Gasolina Aditivada
* Etanol
* Diesel
* Diesel S10
* GNV (Gás Natural Veicular)

Para este estudo, foi selecionado o **período de 2015 a 2025**, por apresentar maior consistência nos dados e abranger eventos econômicos relevantes que impactaram diretamente os preços dos combustíveis, tais como:

* 2016 – Crise econômica no Brasil
* 2018 – Greve dos caminhoneiros (impacto direto no abastecimento)
* 2020 – Pandemia de COVID-19
* 2022 – Alta global dos combustíveis

O conjunto de dados consolidado possui 9.889.848 instâncias (linhas) e 17 atributos (colunas), incluindo variáveis categóricas (como estado, município, produto e bandeira), numéricas contínuas (como valores de compra e venda) e temporais (datas de coleta).

Do ponto de vista analítico, este projeto também investiga a relação entre os preços de venda dos combustíveis ao longo dos anos e o poder de compra da população brasileira, representado pelo **IPCA (Índice Nacional de Preços ao Consumidor Amplo)**, calculado pelo [Banco Central do Brasil](https://www.bcb.gov.br/controleinflacao/historicometas).

O objetivo é verificar a existência de correlação entre a evolução dos preços dos combustíveis e a inflação oficial do país, buscando compreender se os aumentos nos preços acompanham, superam ou divergem da variação inflacionária no período analisado. A análise de correlação permite identificar o grau de associação entre essas variáveis, podendo indicar relações positivas, negativas ou inexistentes.

Ressalta-se que correlação não implica causalidade, ou seja, mesmo que haja associação entre as variáveis, não é possível afirmar que uma seja a causa direta da outra. Ainda assim, essa análise contribui para uma compreensão mais aprofundada do comportamento dos preços dos combustíveis no contexto econômico nacional e seus impactos sobre o poder de compra da população.

Por fim, sob a perspectiva de ciência de dados, o estudo configura-se como um problema de regressão e classificação, tendo como objetivo futuro a previsão dos preços dos combustíveis e a categorização de tendências de mercado (alta, baixa ou estabilidade). Essas aplicações possuem relevância prática, como o apoio à tomada de decisão no setor de transporte e o monitoramento de anomalias no mercado.

---

## 👨‍💻 Integrantes da Equipe

- Bruno Gabriel `<bgprs>`
- Flávia Vitória `<fves>`
- Lucas Cabral `<lsc>`
- Ítallo Augusto `<iapam>`
- Nathan Barbosa `<nbs3>`

---

## 🧱 Tecnologias e Ferramentas

- 🖥️ Linguagem: **Python**
- 💾 Disciplina: **Ciências de Dados e Aprendizagem de Máquina**
- 📋 Documentação: **Google Docs, GitHub, Colab**
- 📞 Comunicação: **Discord, Whatsapp e Github projects**
- 🎨 Apresentação visual: **Canva, Miro**

---

## Repositório

1. Clone o repositório:
```bash
git clone https://github.com/LucasCabra7/precos-combustiveis-brasil-2015-2025.git
```

2. Navegue até o diretório do projeto:
```bash
cd 
```



## 📎 Links Úteis

- 📒 Documentos Google (Documentação): https://docs.google.com/document/d/1OUwCZPyRawydDn3W3Rghq1-GurU5QXb4JByeKr8a1y4/edit?usp=sharing

---

## 📃 Licença

Este projeto é de caráter acadêmico, sem fins lucrativos. Todos os direitos reservados aos autores.

## Equipe do Projeto

<div align="center">

  <table>
    <tr>
      <td align="center">
        <img src="https://avatars.githubusercontent.com/u/162474087?v=4" width="100px" alt="Pessoa 1"/><br/>
        <b>Bruno Ramos 1</b>
      </td>
      <td align="center">
        <img src="https://avatars.githubusercontent.com/u/155683708?v=4" width="100px" alt="Lucas Cabral"/><br/>
        <b>Lucas Cabral</b>
      </td>
      <td align="center">
        <img src="https://avatars.githubusercontent.com/u/147522368?v=4" width="100px"/><br/>
        <b>Itallo Augusto</b>
      </td>
      <td align="center">
        <img src=""/><br/>
        <b></b>
      </td>
      <td align="center">
        <img src="https://avatars.githubusercontent.com/u/205646287?v=4" width="100px" alt="Flavitche"/><br/>
        <b>Flávia Vitória</b>
      </td>
    </tr>
  </table>

</div>

---

<p align="center">
  &copy; 2025 Universidade Federal de Pernambuco - Centro de Informática. Todos os direitos reservados.
</p>

<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=66CDAA&height=120&section=header"/>
