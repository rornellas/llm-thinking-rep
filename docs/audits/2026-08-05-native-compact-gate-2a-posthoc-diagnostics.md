# Diagnóstico pós-hoc — Native Compact Gate 2A

**Status:** análise pós-hoc; não altera o veredito pré-registrado.

**Veredito congelado:** `NATIVE_COMPACT_GATE_2A_FAIL`

## Escala `small`

| Estado/candidato | Cosine entre experts | Variância centrada | Residual/common | Top-1 energia residual | Stable rank residual | Uso do rank nominal |
|---|---:|---:|---:|---:|---:|---:|
| initial/conventional-full | 0.00077 | 0.91596 | — | — | — | — |
| initial/conventional-narrow65 | 0.00047 | 0.91625 | — | — | — | — |
| initial/native-shared-rank | 1.00000 | 0.00000 | 0.00097 | 0.27628 | 3.673 | 45.91% |
| final/conventional-full | 0.00082 | 0.91591 | — | — | — | — |
| final/conventional-narrow65 | 0.00075 | 0.91599 | — | — | — | — |
| final/native-shared-rank | 0.98823 | 0.01087 | 0.10761 | 0.98779 | 1.013 | 12.67% |

- rank-collapse signal: `True`;
- effective-expert-similarity signal: `True`;

## Escala `medium`

| Estado/candidato | Cosine entre experts | Variância centrada | Residual/common | Top-1 energia residual | Stable rank residual | Uso do rank nominal |
|---|---:|---:|---:|---:|---:|---:|
| initial/conventional-full | 0.00004 | 0.93746 | — | — | — | — |
| initial/conventional-narrow65 | -0.00024 | 0.93772 | — | — | — | — |
| initial/native-shared-rank | 1.00000 | 0.00000 | 0.00098 | 0.22955 | 4.408 | 44.08% |
| final/conventional-full | 0.00001 | 0.93749 | — | — | — | — |
| final/conventional-narrow65 | -0.00015 | 0.93763 | — | — | — | — |
| final/native-shared-rank | 0.99065 | 0.00881 | 0.09297 | 0.97769 | 1.028 | 10.28% |

- rank-collapse signal: `True`;
- effective-expert-similarity signal: `True`;

## Interpretação limitada

O diagnóstico mede pesos congelados, não causalidade. Ele mostra se o rank nominal foi utilizado,
mas não prova sozinho se a causa é inicialização, otimização, regularização ou a função do corpus.
A próxima intervenção elegível é um controle simples de utilização dos modos; nested/dynamic rank
e checkpoint real continuam bloqueados.
