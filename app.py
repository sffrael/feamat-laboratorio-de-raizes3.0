import ast
import math
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# FEAMAT — Laboratório de Raízes
# ============================================================

st.set_page_config(
    page_title="FEAMAT | Laboratório de Raízes",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Identidade visual acadêmica azul; compatível com tema claro/escuro.
st.markdown("""
<style>
:root {
  --feamat-blue: #2457c5;
  --feamat-navy: #15346f;
}
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.feamat-hero {
  background: linear-gradient(115deg, #15346f 0%, #2457c5 65%, #3978e8 100%);
  color: white; padding: 2.1rem 2.4rem; border-radius: 22px;
  margin-bottom: 1.5rem; box-shadow: 0 12px 28px rgba(25,65,140,.16);
}
.feamat-hero .eyebrow {font-size:.78rem; letter-spacing:.14em; font-weight:800; opacity:.85;}
.feamat-hero h1 {color:white; font-size:2.7rem; margin:.6rem 0;}
.feamat-hero p {color:#eaf1ff; font-size:1.05rem; margin:0;}
.info-panel {
  border-left: 5px solid #3978e8; border-radius: 10px;
  padding: 1rem 1.1rem; background: rgba(55,120,232,.09);
}
.small-muted {font-size:.9rem; opacity:.8;}
div[data-testid="stMetric"] {
  border: 1px solid rgba(120,140,170,.25); border-radius: 12px;
  padding: .8rem 1rem; background: rgba(120,140,170,.06);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feamat-hero">
  <div class="eyebrow">LABORATÓRIO VIRTUAL DE CÁLCULO NUMÉRICO</div>
  <h1>FEAMAT</h1>
  <p>Laboratório de Raízes · Por trás dos resultados: como os algoritmos numéricos encontram raízes de funções</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Funções predefinidas --------------------
PRESETS = {
    "x² − 2": {
        "expr": "x^2 - 2", "a": 1.0, "b": 2.0, "x0": 1.5,
        "description": "Exemplo clássico: aproximação de √2."
    },
    "x³ − x − 2": {
        "expr": "x^3 - x - 2", "a": 1.0, "b": 2.0, "x0": 1.5,
        "description": "Equação cúbica com uma raiz no intervalo [1, 2]."
    },
    "cos(x) − x": {
        "expr": "cos(x) - x", "a": 0.0, "b": 1.0, "x0": 0.5,
        "description": "Equação transcendental; a raiz está entre 0 e 1."
    },
    "eˣ − 3": {
        "expr": "exp(x) - 3", "a": 0.0, "b": 2.0, "x0": 1.0,
        "description": "A função exponencial encontra a raiz ln(3)."
    },
    "ln(x) − 2": {
        "expr": "log(x) - 2", "a": 7.0, "b": 8.0, "x0": 7.5,
        "description": "Função logarítmica, definida para x positivo."
    },
    "√x − 3": {
        "expr": "sqrt(x) - 3", "a": 8.0, "b": 10.0, "x0": 9.0,
        "description": "Função raiz quadrada; a raiz é 9."
    },
    "Função personalizada": {
        "expr": "x^2 - 2", "a": 1.0, "b": 2.0, "x0": 1.5,
        "description": "Digite sua própria função e configure os parâmetros."
    },
}

# -------------------- Interpretador seguro --------------------
ALLOWED_FUNCTIONS = {
    "sin": math.sin, "sen": math.sin,
    "cos": math.cos,
    "tan": math.tan, "tg": math.tan,
    "exp": math.exp, "sqrt": math.sqrt,
    "log": math.log, "ln": math.log,
    "abs": abs,
}
ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}

def evaluate_node(node, x_value):
    if isinstance(node, ast.Expression):
        return evaluate_node(node.body, x_value)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return float(x_value)
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Nome não permitido: {node.id}")
    if isinstance(node, ast.UnaryOp):
        value = evaluate_node(node.operand, x_value)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.BinOp):
        left = evaluate_node(node.left, x_value)
        right = evaluate_node(node.right, x_value)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow): return left ** right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in ALLOWED_FUNCTIONS or len(node.args) != 1:
            raise ValueError("Função não permitida ou quantidade incorreta de argumentos.")
        return ALLOWED_FUNCTIONS[name](evaluate_node(node.args[0], x_value))
    raise ValueError("Expressão inválida. Use apenas x, operações e funções permitidas.")

def compile_function(expression):
    normalized = expression.replace("^", "**").replace("π", "pi")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Não foi possível interpretar a expressão. Confira a escrita da função.") from exc

    allowed_nodes = (
        ast.Expression, ast.Constant, ast.Name, ast.UnaryOp, ast.UAdd, ast.USub,
        ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Call, ast.Load
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("A expressão contém uma operação não permitida.")
        if isinstance(node, ast.Name):
            if node.id != "x" and node.id not in ALLOWED_CONSTANTS and node.id not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Símbolo não reconhecido: {node.id}")
    def f(x):
        value = evaluate_node(tree, x)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("A função produziu um valor não finito.")
        return value
    return f

def sign_label(value):
    if value > 0: return "Positivo (+)"
    if value < 0: return "Negativo (−)"
    return "Zero (0)"

def sign_symbol(value):
    if value > 0: return "+"
    if value < 0: return "−"
    return "0"

def derivative(f, x):
    h = 1e-6 * max(1.0, abs(x))
    return (f(x + h) - f(x - h)) / (2*h)

# -------------------- Métodos numéricos --------------------
def bisection(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0, 0.0, [], "O extremo a já é uma raiz."
    if fb == 0:
        return b, 0, 0.0, [], "O extremo b já é uma raiz."
    if fa * fb > 0:
        raise ValueError("Não há mudança de sinal em [a, b]. Escolha extremos com f(a) e f(b) de sinais opostos.")

    rows = []
    root = None
    for i in range(1, max_iter + 1):
        m = (a + b) / 2
        fm = f(m)
        if fa * fm < 0:
            new_a, new_b = a, m
            decision = "[a, m] mantido: f(a) e f(m) têm sinais opostos."
        elif fm * fb < 0:
            new_a, new_b = m, b
            decision = "[m, b] mantido: f(m) e f(b) têm sinais opostos."
        else:
            new_a, new_b = m, m
            decision = "O ponto médio é uma raiz exata (dentro da precisão numérica)."

        rows.append({
            "Iteração": i, "a": a, "f(a)": fa, "Sinal f(a)": sign_symbol(fa),
            "b": b, "f(b)": fb, "Sinal f(b)": sign_symbol(fb),
            "m = (a+b)/2": m, "f(m)": fm, "Sinal f(m)": sign_symbol(fm),
            "Intervalo mantido": f"[{new_a:.10g}, {new_b:.10g}]",
            "Decisão": decision
        })
        root = m
        if abs(fm) <= tol or (b - a)/2 <= tol:
            break
        a, b = new_a, new_b
        fa, fb = f(a), f(b)
    residual = abs(f(root))
    return root, len(rows), residual, rows, "Critério de parada atingido." if len(rows) < max_iter else "Limite máximo de iterações atingido."

def newton(f, x0, tol, max_iter):
    rows = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = derivative(f, x)
        if abs(dfx) < 1e-14:
            raise ValueError(f"Derivada numericamente nula na aproximação x = {x:.10g}.")
        x_next = x - fx / dfx
        error = abs(x_next - x)
        rows.append({
            "Iteração": i, "xₙ": x, "f(xₙ)": fx,
            "f'(xₙ) aproximada": dfx, "xₙ₊₁": x_next,
            "Erro |xₙ₊₁ − xₙ|": error
        })
        x = x_next
        if error <= tol or abs(f(x)) <= tol:
            break
    return x, len(rows), abs(f(x)), rows, "Critério de parada atingido." if len(rows) < max_iter else "Limite máximo de iterações atingido."

# -------------------- Barra lateral --------------------
st.sidebar.title("Experimento")
st.sidebar.caption("Escolha uma função ou crie seu próprio experimento.")

preset_names = list(PRESETS.keys())
if "preset_choice" not in st.session_state:
    st.session_state.preset_choice = preset_names[0]
if "func_input" not in st.session_state:
    p = PRESETS[st.session_state.preset_choice]
    st.session_state.func_input = p["expr"]
    st.session_state.param_a = p["a"]
    st.session_state.param_b = p["b"]
    st.session_state.param_x0 = p["x0"]

def load_preset():
    p = PRESETS[st.session_state.preset_choice]
    st.session_state.func_input = p["expr"]
    st.session_state.param_a = p["a"]
    st.session_state.param_b = p["b"]
    st.session_state.param_x0 = p["x0"]

st.sidebar.selectbox(
    "Funções predefinidas", preset_names, key="preset_choice",
    help="Ao selecionar uma função, a expressão e os parâmetros são preenchidos automaticamente.",
    on_change=load_preset
)
current_preset = PRESETS[st.session_state.preset_choice]
st.sidebar.caption(current_preset["description"])

st.sidebar.text_input(
    "Função f(x)", key="func_input",
    help="Use ^ ou ** para potência. Exemplos: x^2-2, cos(x)-x, exp(x)-3."
)
st.sidebar.markdown(
    "<div class='small-muted'>Funções: sin, sen, cos, tan, tg, exp, sqrt, log, ln, abs. "
    "Constantes: pi, e.</div>", unsafe_allow_html=True
)

st.sidebar.subheader("Parâmetros numéricos")
c1, c2 = st.sidebar.columns(2)
with c1:
    st.number_input("Extremo a (Bisseção)", key="param_a", step=0.1, format="%.8f")
with c2:
    st.number_input("Extremo b (Bisseção)", key="param_b", step=0.1, format="%.8f")
st.sidebar.number_input("Valor inicial x₀ (Newton)", key="param_x0", step=0.1, format="%.8f")
tol = st.sidebar.select_slider(
    "Precisão (tolerância)", options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8],
    value=1e-6, format_func=lambda v: f"{v:g}"
)
max_iter = st.sidebar.number_input("Número máximo de iterações", min_value=5, max_value=200, value=50, step=5)

run = st.sidebar.button("▶ Executar comparação", type="primary", use_container_width=True)

# -------------------- Conteúdo principal --------------------
left, right = st.columns([1.35, 1])
with left:
    st.subheader("Investigue como os métodos encontram uma raiz")
    st.write("Informe uma função, configure os parâmetros e acompanhe cada aproximação. Compare Bisseção e Newton-Raphson por meio de tabelas, gráficos e análise do erro.")
with right:
    st.markdown("""
    <div class="info-panel">
      <strong>Como começar</strong><br>
      1. Escolha uma função predefinida ou personalizada.<br>
      2. Confira os parâmetros preenchidos.<br>
      3. Execute a comparação e investigue as iterações.
    </div>
    """, unsafe_allow_html=True)

if run:
    try:
        f = compile_function(st.session_state.func_input)
        a = float(st.session_state.param_a)
        b = float(st.session_state.param_b)
        x0 = float(st.session_state.param_x0)
        if a >= b:
            st.error("O extremo a precisa ser menor que o extremo b.")
            st.stop()

        fa, fb = f(a), f(b)
        st.markdown("---")
        st.subheader("1. Análise inicial dos sinais — Método da Bisseção")
        st.write("A Bisseção exige que os valores nos extremos tenham sinais opostos (ou que um dos extremos já seja raiz).")
        s1, s2, s3 = st.columns(3)
        s1.metric("f(a)", f"{fa:.10g}", sign_label(fa))
        s2.metric("f(b)", f"{fb:.10g}", sign_label(fb))
        opposite = fa * fb < 0
        s3.metric("Comparação", "Sinais opostos" if opposite else ("Extremo é raiz" if fa == 0 or fb == 0 else "Mesmo sinal"),
                  "Intervalo válido" if opposite or fa == 0 or fb == 0 else "Intervalo não válido")

        if fa * fb > 0:
            st.warning("f(a) e f(b) possuem o mesmo sinal. A Bisseção não pode começar nesse intervalo. Escolha outros extremos.")
            bis_result = None
        else:
            st.success("Há mudança de sinal: existe pelo menos uma raiz no intervalo, considerando a continuidade da função.")

            bis_result = bisection(f, a, b, tol, int(max_iter))
            root_b, it_b, res_b, rows_b, msg_b = bis_result
            st.subheader("2. Método da Bisseção: decisões em cada iteração")
            st.caption("Observe os sinais de f(a), f(b) e f(m). O intervalo que contém a mudança de sinal é mantido.")
            df_b = pd.DataFrame(rows_b)
            st.dataframe(df_b, use_container_width=True, hide_index=True)
            st.info(f"Última decisão: {rows_b[-1]['Decisão']}")

            st.subheader("3. Visualização passo a passo da Bisseção")
            if rows_b:
                idx = st.slider("Selecione a iteração", 1, len(rows_b), len(rows_b), key="bis_step") - 1
                row = rows_b[idx]
                st.write(f"**Iteração {row['Iteração']}** — ponto médio: **m = {row['m = (a+b)/2']:.10g}**")
                st.write(f"f(a) = {row['f(a)']:.8g} ({row['Sinal f(a)']}), f(b) = {row['f(b)']:.8g} ({row['Sinal f(b)']}), f(m) = {row['f(m)']:.8g} ({row['Sinal f(m)']}).")
                st.info(row["Decisão"])
                st.caption(f"Novo intervalo: {row['Intervalo mantido']}")

        st.markdown("---")
        st.subheader("4. Método de Newton-Raphson")
        try:
            root_n, it_n, res_n, rows_n, msg_n = newton(f, x0, tol, int(max_iter))
            st.dataframe(pd.DataFrame(rows_n), use_container_width=True, hide_index=True)
            st.caption(msg_n)
        except Exception as exc:
            root_n = it_n = res_n = None
            rows_n = []
            st.error(f"Newton-Raphson: {exc}")

        st.markdown("---")
        st.subheader("5. Comparação dos resultados")
        metric_cols = st.columns(2)
        if bis_result is not None:
            metric_cols[0].metric("Raiz — Bisseção", f"{root_b:.10g}", f"{it_b} iterações")
            metric_cols[0].caption(f"Resíduo |f(x)|: {res_b:.3e}")
        else:
            metric_cols[0].warning("Bisseção não executada: intervalo sem mudança de sinal.")
        if root_n is not None:
            metric_cols[1].metric("Raiz — Newton-Raphson", f"{root_n:.10g}", f"{it_n} iterações")
            metric_cols[1].caption(f"Resíduo |f(x)|: {res_n:.3e}")

        # Gráfico da função
        st.subheader("6. Gráfico da função e aproximações")
        x_min = min(a, b, x0) - 1
        x_max = max(a, b, x0) + 1
        if x_max - x_min < 2:
            x_min, x_max = x_min - 1, x_max + 1
        xs = np.linspace(x_min, x_max, 500)
        ys = []
        for xv in xs:
            try:
                yv = f(float(xv))
                ys.append(yv if abs(yv) < 1e8 else np.nan)
            except Exception:
                ys.append(np.nan)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(xs, ys, label="f(x)")
        ax.axhline(0, linewidth=1, linestyle="--")
        if bis_result is not None:
            ax.scatter([root_b], [f(root_b)], label="Bisseção", zorder=4)
        if root_n is not None:
            ax.scatter([root_n], [f(root_n)], marker="x", s=80, label="Newton-Raphson", zorder=5)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title("Gráfico da função")
        ax.grid(True, alpha=.25)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Exportação
        st.subheader("7. Exportar tabelas")
        export_cols = st.columns(2)
        if bis_result is not None:
            csv_b = df_b.to_csv(index=False).encode("utf-8-sig")
            export_cols[0].download_button("⬇ Baixar tabela da Bisseção (CSV)", csv_b, "feamat_bissecao.csv", "text/csv")
        if root_n is not None:
            csv_n = pd.DataFrame(rows_n).to_csv(index=False).encode("utf-8-sig")
            export_cols[1].download_button("⬇ Baixar tabela de Newton (CSV)", csv_n, "feamat_newton.csv", "text/csv")

    except Exception as exc:
        st.error(f"Não foi possível executar o experimento: {exc}")
else:
    st.markdown("---")
    st.subheader("Comece por um exemplo")
    st.write("Selecione uma das funções predefinidas na barra lateral. O FEAMAT preencherá automaticamente a expressão, os extremos **a** e **b**, e o valor inicial **x₀**.")
    st.markdown("""
    **O que será investigado?**
    - Como a comparação dos sinais de f(a) e f(b) indica se o intervalo pode ser usado na Bisseção.
    - Como o ponto médio divide o intervalo e como a mudança de sinal determina o próximo intervalo.
    - Como Newton-Raphson utiliza a inclinação da função para produzir novas aproximações.
    - Como comparar o número de iterações e o resíduo de cada método.
    """)
    st.info("Dica: comece com f(x) = x² − 2, no intervalo [1, 2], e valor inicial x₀ = 1,5.")
