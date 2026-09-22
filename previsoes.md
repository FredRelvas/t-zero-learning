# Previsões — Parte 3

Escritas **antes** de rodar as varreduras, conforme o passo 1 do protocolo da atividade.

Baseline de referência (seed 1, config padrão, 500k passos):
`episodic_return_mean_last100` = 477,84 · `eval/mean_return` = 500 ± 0 ·
`losses/q_values` = 105,79 · `losses/td_loss` = 0,057.

## Q1 — Frequência de sincronização da target network

Espero um trade-off entre estabilidade e velocidade de propagação do valor. Com
`target_network_frequency=1`, o alvo é a própria rede que está sendo treinada: ao empurrar
Q(s,a) para cima, Q(s′) sobe junto, o alvo "foge" enquanto o perseguimos, e o `max` acumula
sobrestimações — prevejo retorno instável e `q_values` acima do teto teórico de
1/(1−γ) = 100. Com `target_network_frequency=50000`, o alvo fica estável mas desatualizado,
e como cada sincronização propaga apenas um passo de bootstrap — c(k) = (1 − γᵏ)/(1 − γ),
com apenas 9 syncs no treino inteiro — prevejo `q_values` subindo em degraus até a ordem de
10, bem abaixo do valor real, com `td_loss` em dente de serra (quase zero entre syncs, pico
a cada sync). Esse segundo caso deve ser justamente aquele em que a loss parece ótima
enquanto os valores estão errados, porque ela mede a distância até o alvo, não até o valor
verdadeiro.

## Q2 — Tamanho do replay buffer

Espero que o buffer pequeno (500) falhe por dois motivos distintos. O primeiro é
correlação: com `batch_size=128`, cada minibatch é cerca de um quarto do buffer inteiro, de
modo que as amostras vêm quase todas da mesma trajetória e os gradientes sucessivos apontam
na mesma direção, elevando a variância das atualizações. O segundo é cobertura: conforme os
episódios se alongam, 500 transições passam a guardar pouco mais de um episódio da política
atual, então a rede nunca revê os estados que visitava antes e esquece o que já havia
aprendido sobre eles. Prevejo, portanto, `q_values` oscilando e curva de retorno que sobe e
colapsa ciclicamente, em vez de degradar de forma suave. Já com 100000 o buffer guarda dados
de políticas bem antigas, o que deve atrasar um pouco a subida inicial; como o alvo de
Q-learning usa `max` e não depende da política que coletou os dados, espero desempenho final
próximo do baseline.

## Q3 (extra) — Fração de exploração

Gráficos que escolhi reportar: `charts/epsilon`, `charts/episodic_return_mean_last100` e
`eval/mean_return`. A escolha é o próprio argumento: `epsilon` mostra o cronograma
efetivamente usado, o retorno episódico mede a política **epsilon-greedy que coleta os
dados**, e o `eval/mean_return` mede a política **greedy** ao final — e a previsão é que
esses dois últimos discordem. Com `exploration_fraction=0.05`, o epsilon cai de 1,0 a 0,05
já em 25 mil passos, quase junto com o início do aprendizado (`learning_starts=10000`);
espero subida rápida do retorno, com algum risco de convergência prematura por o buffer se
estreitar cedo demais. Com `exploration_fraction=0.95`, o epsilon só chega a 0,05 em 475 mil
passos e fica em torno de 0,5 na metade do treino, então metade das ações é aleatória e o
bastão cai cedo: prevejo a curva de retorno presa em valores baixos durante quase todo o
treino, com um salto abrupto perto do fim, quando o epsilon finalmente desce — e mesmo assim
um `eval/mean_return` alto, porque a política greedy aprendida é melhor do que a curva de
treino sugere.
