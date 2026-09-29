# Previsões — A2C, Parte 4

Escritas **antes** de rodar as varreduras, conforme o passo 1 do protocolo da atividade.

Baseline de referência (seed 1, config padrão, 500k passos, 29 s):
`episodic_return_mean_last100` = 470,79 · `eval/mean_return` = 500 ± 0 ·
`losses/entropy` = 0,56 · `losses/explained_variance` = −0,42 · `charts/SPS` = 19.540.
Lote = `num_envs` × `num_steps` = 8 × 5 = 40, o que dá 12.500 atualizações em 500k passos.

Varreduras planejadas: Q1 `num_envs` ∈ {1, 8, 32}; Q2 `a2c.num_steps` ∈ {1, 5, 32, 128};
Q3 `a2c.ent_coef` ∈ {0, 0,01, 0,1} com 2 seeds em 0; Q4 `a2c.use_baseline=false` com 2 seeds.

## Q1 — Número de atores em paralelo

Espero que `num_envs` misture dois efeitos opostos e que o baseline esteja perto do ponto
ótimo. Com 1 ambiente o lote cai para 5 transições *consecutivas do mesmo episódio* — quase
uma amostra só —, então prevejo um `policy_loss` com amplitude de oscilação várias vezes
maior que a do baseline: é o ruído de um gradiente calculado sobre dados correlacionados, e
é exatamente o argumento que o A3C usa para justificar vários atores num método on-policy
que não pode ter replay buffer. O outro efeito puxa para o lado contrário: com lote de 5 o
treino faz 100.000 atualizações em vez de 12.500, e com 32 ambientes faz apenas 3.125 — então
prevejo que `num_envs=1` aprenda de forma lenta e instável apesar de *mais* passos de
gradiente, e que 32 tenha o gradiente mais limpo da varredura mas comece a mostrar sinais de
subtreino dentro do orçamento fixo de 500k passos.
No `SPS` espero crescimento claro de 1 para 8 e saturação depois, porque a máquina tem 10
núcleos e a partir daí o custo de coordenar subprocessos come o ganho.

## Q2 — Horizonte do retorno de n passos

Espero ver o trade-off viés-variância aparecer separadamente no crítico e na política. Com
`num_steps=1` o alvo é `r + γV(s′)`, quase todo bootstrap: pouquíssima variância e muito
viés, então prevejo `value_loss` muito pequeno e — aqui está a armadilha — uma
`explained_variance` enganosamente alta, possivelmente perto de 1, porque o alvo passa a ser
quase uma função determinística de V no estado seguinte e a métrica acaba medindo
*consistência de um passo* em vez de acerto do valor verdadeiro. Com `num_steps=128` o alvo
é quase Monte Carlo: prevejo `value_loss` uma ou duas ordens de grandeza maior pela variância
das somas longas, e um retorno ruim principalmente por subtreino, já que o lote de 1.024 deixa
só 488 atualizações no orçamento de 500k passos. Prevejo que o baseline de 5 vença os dois
extremos no retorno final, e que o ator sofra junto com o crítico nos extremos porque a
vantagem `R − V(s)` é o peso do gradiente de política: crítico viesado ou ruidoso vira peso
errado, não apenas um gráfico feio.

## Q3 — Coeficiente de entropia

Espero que o bônus de entropia seja o único freio contra a política virar determinística cedo
demais, já que no A2C não existe epsilon-greedy e toda a exploração vem de amostrar a
`Categorical`. Com `ent_coef=0` prevejo a entropia caindo rápido de ln 2 ≈ 0,69 para perto de
zero, mas o CartPole ainda sendo resolvido na maioria das seeds — só duas ações, recompensa
densa e positiva em todo passo, e uma política gulosa razoável é fácil de achar; espero que a
diferença entre as duas seeds seja bem maior que no baseline, e é por isso que rodo as duas.
Com `ent_coef=0,1` prevejo o efeito oposto e mais dramático: como o crítico melhora ao longo
do treino e as vantagens encolhem, o termo de entropia (constante) passa a dominar o gradiente,
a entropia fica presa perto de 0,69 e o retorno estaciona bem abaixo de 500 — o agente é pago
para permanecer indeciso.

## Q4 — Ablação do baseline

Espero confirmar nos gráficos o resultado teórico de que subtrair o baseline não muda o
gradiente esperado, só a variância. Prevejo `advantage_mean` oscilando em torno de zero no
baseline e fortemente positivo sem ele — o retorno de 5 passos com γ = 0,99 vale
≈ 4,9 + 0,95·V(s), então da ordem de dezenas —, com `advantage_std` também muito maior, já que
sem centrar sobra a variação de valor *entre estados*, que nada tem a ver com a qualidade da
ação. A consequência que espero ver na entropia: com peso sempre positivo, **toda** ação
amostrada é reforçada, e como as ações são amostradas na proporção da probabilidade atual
surge uma realimentação de "rico fica mais rico" que colapsa a entropia mais rápido que no
baseline, sem nenhuma relação com a ação ser boa. Prevejo ainda um efeito colateral de
implementação: com pesos da ordem de dezenas contra ~1 do baseline, o `max_grad_norm=0,5`
deve clipar praticamente todo passo, de modo que parte da degradação medida virá do clipping
e não só da variância.
