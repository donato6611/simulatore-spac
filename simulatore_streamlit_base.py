import streamlit as st
import pandas as pd

import numpy as np

# --- Funzione per il grafico riepilogativo delle offerte UP (versione corretta e unica) ---
def plot_offers_summary(df):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    # Dimensione dinamica e font adattivi migliorati
    n_up = len(df)
    width = min(max(8, 1.18*n_up), 12.5)
    font_scale = max(0.88, min(1.10, width/11))
    height = 3.5 + 1.5*font_scale
    fig, ax1 = plt.subplots(figsize=(width, height))
    color_map = {"FCMT": "#43c07a", "FCMNT": "#457b9d"}
    # Istogramma delle offerte (quantità) per UP
    bars = ax1.bar(
        df["UP"],
        df["q (MWh)"],
        color=[color_map.get(t, "gray") for t in df["Tipo"]],
        edgecolor='k',
        alpha=0.78,
        zorder=2
    )
    min_font = 11
    ax1.set_ylabel("Quantità offerta (MWh)", fontsize=max(int(18*font_scale), min_font), labelpad=int(8*font_scale))
    ax1.set_xlabel("UP", fontsize=max(int(18*font_scale), min_font), labelpad=int(8*font_scale))
    # Titolo: font armonizzato rispetto alla legenda (solo leggermente più grande)
    ax1.set_title(
        "Offerte UP: quantità (barre), prezzo (linea/arancione), tipo (colore)",
        fontsize=max(int(16*font_scale), min_font+6), pad=int(12*font_scale)
    )
    # Etichette asse x e y
    ax1.tick_params(axis='x', labelsize=max(int(13*font_scale), min_font), pad=int(4*font_scale))
    ax1.tick_params(axis='y', labelsize=max(int(13*font_scale), min_font), pad=int(4*font_scale))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.35, zorder=1)
    # Asse secondario per il prezzo offerto
    ax2 = ax1.twinx()
    ax2.plot(
        df["UP"], df["p (€/MWh)"],
        color="#e76f51", marker="o", linewidth=2.2*font_scale,
        markersize=max(10*font_scale, 7), markeredgewidth=1.5*font_scale, markeredgecolor="#222",
        label="Prezzo offerto (€/MWh)", zorder=3
    )
    ax2.set_ylabel("Prezzo offerto (€/MWh)", fontsize=max(int(14*font_scale), min_font), color="#e76f51", labelpad=int(7*font_scale))
    ax2.tick_params(axis='y', labelcolor="#e76f51", labelsize=max(int(12*font_scale), min_font), pad=int(4*font_scale))
    # Etichette prezzo sopra i marker (ruotate se tanti UP)
    rotate_labels = n_up > 8
    for i, up in enumerate(df["UP"]):
        p = df.iloc[i]["p (€/MWh)"]
        ax2.text(
            i, p+4.5*font_scale, f"{p:.0f}", ha='center', va='bottom',
            fontsize=max(int(13*font_scale), min_font), color="#e76f51", fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='#e76f51', boxstyle='round,pad=0.13', alpha=0.7), zorder=4,
            clip_on=True,
            rotation=30 if rotate_labels else 0
        )

    # Etichette sopra le barre: solo quantità (senza unità di misura)
    for i, bar in enumerate(bars):
        tipo = df.iloc[i]["Tipo"]
        ax1.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() - 1.2*font_scale,
            f"{bar.get_height():.1f}",
            ha='center', va='bottom', fontsize=max(int(12*font_scale), min_font), fontweight='bold',
            color=color_map.get(tipo, "gray"),
            path_effects=[pe.withStroke(linewidth=max(2.2*font_scale, 1.2), foreground="white")],
            clip_on=True
        )

    # Limiti asse y1 e y2 per non far uscire le etichette
    max_q = max(df["q (MWh)"])
    max_p = max(df["p (€/MWh)"])
    ax1.set_ylim(0, max_q*1.18+6*font_scale)
    ax2.set_ylim(0, max_p*1.18+7*font_scale)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.91, bottom=0.15)

    # Legenda compatta e moderna
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='FCMT', markerfacecolor=color_map['FCMT'], markeredgecolor='k', markersize=max(13*font_scale, 8)),
        Line2D([0], [0], marker='s', color='w', label='FCMNT', markerfacecolor=color_map['FCMNT'], markeredgecolor='k', markersize=max(13*font_scale, 8)),
        Line2D([0], [0], marker='o', color='#e76f51', label='Prezzo offerto', markerfacecolor='#e76f51', markeredgecolor='#222', markersize=max(10*font_scale, 6))
    ]
    legend_fontsize = max(int(13*font_scale), min_font+2)
    legend_title_fontsize = legend_fontsize + 2
    ax1.legend(
        handles=legend_elements,
        title="Tipo UP / Prezzo",
        loc='upper right',
        bbox_to_anchor=(1, -0.18),
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        borderpad=1.0,
        labelspacing=0.8,
        handletextpad=0.5,
        borderaxespad=0.5,
        frameon=True,
        fancybox=True,
        edgecolor="#bbb",
        facecolor=(1,1,1,0.85)
    )
    plt.tight_layout(pad=0.6)
    return fig

st.set_page_config(page_title="SpACapp - Simulatore Base Disaccoppiamento", layout="centered")
st.title("SPaCapp - Simulatore Base Disaccoppiamento Mercato Elettrico con Segmented Pay as Clear (SPaC)")
st.markdown("""
**Legenda:**
- **UP**: Unità di Produzione (centrale, impianto o gruppo che offre energia sul mercato)
- **FCMT**: Fonti con Costi Marginali Trascurabili (ad esempio rinnovabili o nucleare)
- **FCMNT**: Fonti con Costi Marginali Non Trascurabili (ad esempio fossili)
- **q (MWh)**: Quantità di energia offerta
- **p (€/MWh)**: Prezzo dell'energia offerto    
- **Domanda totale (MWh)**: Quantità di energia totale richiesta dal mercato   

Simulazione del clearing classico e disaccoppiato con visualizzazione delle UP accettate/escluse e delle curve domanda/offerta per FCMT e FCMNT.
""")

def get_default_offers():
    # Cambia intestazioni colonne per includere unità di misura
    return pd.DataFrame([
        {"UP": "UP1", "q (MWh)": 5, "p (€/MWh)": 50, "Tipo": "FCMT"},
        {"UP": "UP2", "q (MWh)": 5, "p (€/MWh)": 60, "Tipo": "FCMT"},
        {"UP": "UP3", "q (MWh)": 4, "p (€/MWh)": 160, "Tipo": "FCMT"},
        {"UP": "UP4", "q (MWh)": 6, "p (€/MWh)": 70, "Tipo": "FCMT"},
        {"UP": "UP5", "q (MWh)": 3, "p (€/MWh)": 90, "Tipo": "FCMT"},
        {"UP": "UP6", "q (MWh)": 5, "p (€/MWh)": 190, "Tipo": "FCMNT"},
        {"UP": "UP7", "q (MWh)": 5, "p (€/MWh)": 220, "Tipo": "FCMNT"},
        {"UP": "UP8", "q (MWh)": 7, "p (€/MWh)": 250, "Tipo": "FCMNT"},
        {"UP": "UP9", "q (MWh)": 6, "p (€/MWh)": 200, "Tipo": "FCMNT"},
        {"UP": "UP10", "q (MWh)": 3, "p (€/MWh)": 230, "Tipo": "FCMNT"},
    ])



# --- Sidebar: gestione UP ---
# (Funzionalità di aggiunta/eliminazione UP rimossa su richiesta utente)

def clear_market(offers, demand):
    offers = sorted(offers, key=lambda x: x['p'])
    accepted = []
    total = 0
    marginal_price = 0
    for o in offers:
        if total + o['q'] < demand:
            accepted.append({'UP': o['UP'], 'q': o['q'], 'p': o['p']})
            total += o['q']
        else:
            accepted.append({'UP': o['UP'], 'q': demand - total, 'p': o['p']})
            marginal_price = o['p']
            total = demand
            break
    if marginal_price == 0 and accepted:
        marginal_price = accepted[-1]['p']
    return accepted, marginal_price

def decoupled_clearing(offers, demand, step=0.1):
    best_cost = float('inf')
    best_split = None
    best_prices = None
    best_acc = None
    for d_fcmt in np.arange(0, demand+step, step):
        d_fcmnt = demand - d_fcmt
        fcmt_offers = [o for o in offers if o['type'] == 'FCMT']
        fcmnt_offers = [o for o in offers if o['type'] == 'FCMNT']
        if sum(o['q'] for o in fcmt_offers) < d_fcmt or sum(o['q'] for o in fcmnt_offers) < d_fcmnt:
            continue
        acc_fcmt, p_fcmt = clear_market(fcmt_offers, d_fcmt) if d_fcmt > 0 else ([], 0)
        acc_fcmnt, p_fcmnt = clear_market(fcmnt_offers, d_fcmnt) if d_fcmnt > 0 else ([], 0)
        cost = d_fcmt * p_fcmt + d_fcmnt * p_fcmnt
        if cost < best_cost:
            best_cost = cost
            best_split = (d_fcmt, d_fcmnt)
            best_prices = (p_fcmt, p_fcmnt)
            best_acc = (acc_fcmt, acc_fcmnt)
    return best_split, best_prices, best_cost, best_acc

def plot_up_status(offers, acc_fcmt, acc_fcmnt):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    fig, ax = plt.subplots(figsize=(8,2))
    # FCMT su una riga, FCMNT su un'altra, con separazione e markers diversi
    fcmt_offers = [o for o in offers if o['type']=='FCMT']
    fcmnt_offers = [o for o in offers if o['type']=='FCMNT']
    # FCMT: marker verde (accettata) o rosso (esclusa)
    for i, o in enumerate(fcmt_offers):
        acc = next((a for a in acc_fcmt if a['UP']==o['UP'] and a['q']>0), None)
        color = '#43c07a' if acc else '#e76f51'
        marker = 'o' if acc else 'x'
        text_color = 'black'
        ax.scatter(i, 1, s=400, color=color, marker=marker, edgecolor='k', zorder=3)
        # Etichetta sopra il marker (posizione ottimale, come prima: 1.23)
        ax.text(i, 1.23, o['UP'], ha='center', fontsize=10, color=text_color, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    # FCMNT: marker blu (accettata) o arancio (esclusa)
    for i, o in enumerate(fcmnt_offers):
        acc = next((a for a in acc_fcmnt if a['UP']==o['UP'] and a['q']>0), None)
        color = '#457b9d' if acc else '#f4a261'
        marker = 'o' if acc else 'x'
        text_color = 'black'
        ax.scatter(i, 0, s=400, color=color, marker=marker, edgecolor='k', zorder=3)
        # Etichetta più staccata sotto il marker (abbassata rispetto a prima: -0.28 -> -0.38)
        ax.text(i, -0.38, o['UP'], ha='center', fontsize=10, color=text_color, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.set_yticks([0,1])
    ax.set_yticklabels(['FCMNT','FCMT'])
    ax.set_xticks([])
    ax.set_xlim(-0.5, max(len(fcmt_offers),len(fcmnt_offers))-0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('UP accettate (verde/blu) ed escluse (rosso/arancio)')
    ax.axis('off')
    # Legenda marker e colori
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='FCMT accettata', markerfacecolor='#43c07a', markersize=15, markeredgecolor='k'),
        Line2D([0], [0], marker='x', color='w', label='FCMT esclusa', markerfacecolor='#e76f51', markeredgecolor='#e76f51', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='FCMNT accettata', markerfacecolor='#457b9d', markersize=15, markeredgecolor='k'),
        Line2D([0], [0], marker='x', color='w', label='FCMNT esclusa', markerfacecolor='#f4a261', markeredgecolor='#f4a261', markersize=15)
    ]
    # Sposta la legenda molto più in basso (ad esempio -0.85)
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.65), ncol=2, fontsize=12, frameon=False)
    return fig

# --- INIZIO BLOCCO SIMULAZIONE BASE (come simulatore_streamlit.py, con aggiunte grafiche) ---


# --- Sidebar: controlli animazione ottimizzazione ---
st.sidebar.header("Animazione ottimizzazione SPaC")
run_optimization = st.sidebar.checkbox("Esegui ottimizzazione passo-passo", value=False)
step_opt = st.sidebar.slider("Passo ottimizzazione (MWh)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
speed_opt = st.sidebar.slider("Velocità animazione (secondi)", min_value=0.0, max_value=2.0, value=0.3, step=0.05)
go_to_optimum = st.sidebar.button("Vai direttamente al risultato ottimizzato")

st.header("1. IMPOSTA LE OFFERTE DELLE UNITA DI PRODUZIONE UP E LA DOMANDA TOTALE")
st.markdown("""
<span style='color:#2a9d8f'><b>Per modificare una UP, clicca direttamente sulla cella che vuoi cambiare, digita il dato e premi invio o usa i pulsanti + e -streamlit run simulatore_streamlit_base.py.<br>
<span style='color:#e76f51'>L'aggiunta o la rimozione di UP non è consentita in questa versione.</span></b></span>
""", unsafe_allow_html=True)

# Inizializza la tabella UP solo la prima volta
if 'offers_df' not in st.session_state:
    st.session_state['offers_df'] = get_default_offers()



# Editor tabella UP: solo modifica celle, nessuna aggiunta/rimozione
tipo_options = ["FCMT", "FCMNT"]
edited_df = st.session_state['offers_df'].copy()


for idx in edited_df.index:
    up_name = str(edited_df.at[idx, "UP"])
    col1, col2, col3, col4, col5 = st.columns([2,2,2,2,1])
    with col1:
        st.write(up_name)
    with col2:
        edited_df.at[idx, "q (MWh)"] = st.number_input(
            f"q (MWh) {up_name}",
            min_value=0.0,
            value=float(edited_df.at[idx, "q (MWh)"]),
            step=0.1,
            key=f"q_{up_name}")
    with col3:
        edited_df.at[idx, "p (€/MWh)"] = st.number_input(
            f"p (€/MWh) {up_name}",
            min_value=0.0,
            value=float(edited_df.at[idx, "p (€/MWh)"]),
            step=1.0,
            key=f"p_{up_name}")
    with col4:
        edited_df.at[idx, "Tipo"] = st.radio(
            f"Tipo {up_name}",
            tipo_options,
            index=tipo_options.index(edited_df.at[idx, "Tipo"]),
            horizontal=True,
            key=f"tipo_{up_name}")
    with col5:
        # Mostra l'icona corrispondente al tipo
        tipo = edited_df.at[idx, "Tipo"]
        if tipo == "FCMT":
            st.markdown("<span style='font-size:2em;' title='Fonti rinnovabili o nucleare (fotovoltaico)'>🔆</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='font-size:2em;' title='Centrale termoelettrica'>🏭</span>", unsafe_allow_html=True)

df = edited_df


# Rimuovi eventuali duplicati sulla colonna UP e resetta l'indice

if df['UP'].duplicated().any():
    st.error('Attenzione: sono presenti UP duplicate! Ogni UP deve avere un nome univoco.')
    df = df.drop_duplicates(subset=['UP']).reset_index(drop=True)

# --- Grafico riepilogativo offerte UP ---
st.subheader("Riepilogo offerte delle UP")
st.pyplot(plot_offers_summary(df))



# RIMOSSA LA DOPPIA BARRA DELLA DOMANDA



# Barra per la domanda totale (una sola volta, subito dopo l'intestazione)
st.subheader("Domanda totale (MWh)")
if 'demand' not in st.session_state:
    st.session_state['demand'] = 23.7
demand = st.number_input("Domanda totale (MWh)", min_value=1.0, value=st.session_state['demand'], step=0.1, key="domanda_totale")

# Simulazione: usa sempre df corrente
# Conversione colonne per compatibilità con funzioni esistenti
offers = [
    {
        "UP": row["UP"],
        "q": float(row["q (MWh)"]),
        "p": float(row["p (€/MWh)"]),
        "type": row["Tipo"]
    }
    for _, row in df.iterrows()
]

# Controllo se la domanda supera la somma delle offerte
total_supply = sum(o['q'] for o in offers)
if demand > total_supply:
    st.error(f"La somma delle quantità offerte dalle UP ({total_supply} MWh) è minore della domanda ({demand} MWh): il mercato non ha soluzione!")
    st.stop()
accepted_classic, marginal_price_classic = clear_market(offers, demand)
classic_cost = demand * marginal_price_classic

# --- Animazione ottimizzazione SPaC ---
import time

def decoupled_clearing_animated(offers, demand, step=0.1, speed=0.2, animate=False, go_optimum=False):
    best_cost = float('inf')
    best_split = None
    best_prices = None
    best_acc = None
    history = []
    progress_placeholder = st.empty() if animate else None
    fig_placeholder = st.empty() if animate else None
    for d_fcmt in np.arange(0, demand+step, step):
        d_fcmnt = demand - d_fcmt
        fcmt_offers = [o for o in offers if o['type'] == 'FCMT']
        fcmnt_offers = [o for o in offers if o['type'] == 'FCMNT']
        if sum(o['q'] for o in fcmt_offers) < d_fcmt or sum(o['q'] for o in fcmnt_offers) < d_fcmnt:
            continue
        acc_fcmt, p_fcmt = clear_market(fcmt_offers, d_fcmt) if d_fcmt > 0 else ([], 0)
        acc_fcmnt, p_fcmnt = clear_market(fcmnt_offers, d_fcmnt) if d_fcmnt > 0 else ([], 0)
        cost = d_fcmt * p_fcmt + d_fcmnt * p_fcmnt
        history.append((d_fcmt, d_fcmnt, p_fcmt, p_fcmnt, cost, acc_fcmt, acc_fcmnt))
        if cost < best_cost:
            best_cost = cost
            best_split = (d_fcmt, d_fcmnt)
            best_prices = (p_fcmt, p_fcmnt)
            best_acc = (acc_fcmt, acc_fcmnt)
        # Animazione: aggiorna grafico e info
        if animate and not go_optimum:
            with fig_placeholder.container():
                st.markdown(f"**Domanda FCMT:** {d_fcmt:.1f} MWh a {p_fcmt} €/MWh  ")
                st.markdown(f"**Domanda FCMNT:** {d_fcmnt:.1f} MWh a {p_fcmnt} €/MWh  ")
                st.markdown(f"**Costo totale attuale:** {cost:.0f} €")
                st.pyplot(plot_costs_animation(classic_cost, cost, best_cost))
            progress = 100 * (d_fcmt / demand) if demand > 0 else 0
            progress_placeholder.progress(min(int(progress), 100), text=f"Ottimizzazione: {progress:.1f}%")
            time.sleep(speed)
    if animate and not go_optimum:
        progress_placeholder.empty()
        fig_placeholder.empty()
    return best_split, best_prices, best_cost, best_acc, history

# Funzione per grafico animato costi
def plot_costs_animation(classic_cost, current_cost, best_cost):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,4))
    labels = ['PaC Classico', 'Iterazione attuale', 'Migliore']
    costi = [classic_cost, current_cost, best_cost]
    colors = ['#888', '#f4a261', '#2a9d8f']
    bars = ax.bar(labels, costi, color=colors)
    for i, v in enumerate(costi):
        ax.text(i, v+50, f"{v:.0f}", ha='center')
    ax.set_ylabel('Costo Totale (€)')
    ax.set_title('Confronto costo sistema (animazione)')
    # Legenda fuori dal grafico: la mostreremo con st.markdown
    return fig

# --- Logica per animazione o salto diretto ---
if run_optimization and not go_to_optimum:
    st.info("Animazione ottimizzazione in corso...")
    # Legenda esplicativa fuori dal grafico
    st.markdown("""
    <div style='background-color:#f8f9fa; border-radius:8px; padding:10px; margin-bottom:10px;'>
    <b>Legenda grafico ottimizzazione:</b><br>
    <span style='color:#888'><b>Grigio</b></span>: costo totale con mercato unico (PaC classico)<br>
    <span style='color:#f4a261'><b>Arancione</b></span>: costo totale per la combinazione attuale FCMT/FCMNT<br>
    <span style='color:#2a9d8f'><b>Verde</b></span>: miglior costo trovato finora con combinazioni FCMT/FCMNT (ottimo parziale)
    </div>
    """, unsafe_allow_html=True)
    split, prices, cost, acc, history = decoupled_clearing_animated(offers, demand, step=step_opt, speed=speed_opt, animate=True, go_optimum=False)
elif go_to_optimum:
    split, prices, cost, acc, history = decoupled_clearing_animated(offers, demand, step=step_opt, speed=0, animate=False, go_optimum=True)

else:
    split, prices, cost, acc = decoupled_clearing(offers, demand, step=step_opt)
    history = None
acc_fcmt, acc_fcmnt = acc
# Calcolo prezzo medio ponderato per SPaC (mercato segmentato) DOPO la definizione di split e prices
if 'split' in locals() and 'prices' in locals() and split and prices and (split[0] + split[1] > 0):
    weighted_price_spac = (split[0] * prices[0] + split[1] * prices[1]) / (split[0] + split[1])
else:
    weighted_price_spac = 0




st.header("2. RISULTATI DELLA SIMULAZIONE: CONFRONTO TRA CLEARING CLASSICO E SPaC")


# --- Tabella di confronto risultati aggiornata ---
tabella_confronto = pd.DataFrame({
    "": [
        "Prezzo unitario (€/MWh)",
        "Costo totale (€)",
        "Domanda totale (MWh)",
        "Domanda FCMT (MWh)",
        "Prezzo FCMT (€/MWh)",
        "Domanda FCMNT (MWh)",
        "Prezzo FCMNT (€/MWh)"
    ],
    "Clearing classico": [
        f"{marginal_price_classic:.2f}",
        f"{classic_cost:.0f}",
        f"{demand:.2f}",
        "-",
        "-",
        "-",
        "-"
    ],
    "SPaC": [
        f"{weighted_price_spac:.2f}",
        f"{cost:.0f}",
        f"{demand:.2f}",
        f"{split[0]:.2f}",
        f"{prices[0]:.2f}",
        f"{split[1]:.2f}",
        f"{prices[1]:.2f}"
    ]
})
st.table(tabella_confronto.set_index("").style.set_properties(**{"text-align": "center"}))

# --- Blocco riepilogo risparmi ---

# Arrotonda il risparmio in euro all'intero più vicino
risparmio_euro = int(round(classic_cost - cost))
st.markdown("""
<div style='background-color:#e9f5ec; border-radius:8px; padding:12px; margin-top:10px; margin-bottom:10px;'>
<b>Risparmi ottenuti con SPaC rispetto al classico:</b><br>
<ul style='margin-bottom:0;'>
  <li><b>Risparmio sul costo totale:</b> <span style='color:#2a9d8f; font-size:1.1em'><b>{} €</b></span> (<b>{:.1f}%</b>)</li>
  <li><b>Risparmio sul prezzo medio esitato:</b> <span style='color:#457b9d; font-size:1.1em'><b>{:.2f} €/MWh</b></span> (<b>{:.1f}%</b>)</li>
</ul>
</div>
""".format(
    risparmio_euro,
    100*(classic_cost-cost)/classic_cost if classic_cost else 0,
    marginal_price_classic - weighted_price_spac,
    100*(marginal_price_classic-weighted_price_spac)/marginal_price_classic if marginal_price_classic else 0
), unsafe_allow_html=True)

# Grafico costi (come simulatore_streamlit.py)

# Grafico costi e prezzi medi
def plot_costs_and_prices():
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(6,4))
    labels = ['PaC Classico', 'SPaC Ottimale']
    costi = [classic_cost, cost]
    prezzi = [marginal_price_classic, weighted_price_spac]
    color_cost = ['#888', '#2a9d8f']
    bars = ax1.bar(labels, costi, color=color_cost, width=0.4, label='Costo Totale (€)')
    # Etichette del costo totale con colore corrispondente alla barra
    for i, v in enumerate(costi):
        colore = color_cost[i]
        ax1.text(i-0.15, v+50, f"{v:.0f}", ha='center', fontsize=10, color=colore, fontweight='bold')
    ax1.set_ylabel('Costo Totale (€)', color='#222')
    ax1.set_title('Confronto costo sistema e prezzo finale')

    # Secondo asse per i prezzi
    ax2 = ax1.twinx()
    # Linea più spessa, colore più acceso, marker grande e bordo
    ax2.plot(labels, prezzi, color='#e76f51', marker='o', markersize=12, markeredgewidth=2, markeredgecolor='#222', linewidth=3, label='Prezzo finale (€/MWh)', zorder=10)
    for i, v in enumerate(prezzi):
        ax2.text(i+0.15, v+2, f"{v:.2f}", ha='center', color='#e76f51', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', edgecolor='#e76f51', boxstyle='round,pad=0.2', alpha=0.8))
    ax2.set_ylabel('Prezzo finale (€/MWh)', color='#e76f51', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(prezzi)*1.3)

    # Scritta "Risparmio SPaC x %" sopra la barra verde (SPaC)
    risparmio_perc = 100 * (classic_cost - cost) / classic_cost if classic_cost else 0
    testo_risparmio = f"Risparmio SPaC {risparmio_perc:.1f}%"
    # Posiziona la scritta poco sopra la barra dello SPaC (seconda barra, indice 1)
    y_spac = cost + max(costi)*0.04  # 4% sopra la barra
    ax1.text(1, y_spac, testo_risparmio, ha='center', va='bottom', fontsize=12, color='#2a9d8f', fontweight='bold')

    # Legenda personalizzata sotto il grafico a destra, con font più piccolo
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#888', edgecolor='k', label='Costo Totale PaC Classico'),
        Patch(facecolor='#2a9d8f', edgecolor='k', label='Costo Totale SPaC'),
        Line2D([0], [0], marker='o', color='#e76f51', label='Prezzo finale (€/MWh)', markerfacecolor='#e76f51', markeredgecolor='#222', markersize=11, linewidth=0)
    ]
    legend_fontsize = 8
    legend_title_fontsize = 9
    ax1.legend(
        handles=legend_elements,
        title="Legenda",
        loc='upper right',
        bbox_to_anchor=(1, -0.13),  # più vicino all'asse x
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        borderpad=0.6,
        labelspacing=0.5,
        handletextpad=0.3,
        borderaxespad=0.3,
        frameon=True,
        fancybox=True,
        edgecolor="#bbb",
        facecolor=(1,1,1,0.85)
    )
    return fig

st.pyplot(plot_costs_and_prices())



# Visualizzazione UP accettate/escluse (grafica simpatica, aggiornata)
st.subheader("UP accettate ed escluse nei due mercati previsti nello SPaC")
st.pyplot(plot_up_status(offers, acc_fcmt, acc_fcmnt))

# Curve domanda/offerta FCMT e FCMNT (aggiornate)
st.subheader("Curve domanda/offerta mercati FCMT e FCMNT")
def plot_supply_demand_curve(offers, acc, demand, market_type):
    offers = [o for o in offers if o['type']==market_type]
    offers = sorted(offers, key=lambda x: x['p'])
    q = [o['q'] for o in offers]
    p = [o['p'] for o in offers]
    up_labels = [o['UP'] for o in offers]
    q_cum = np.cumsum(q)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    # Curva offerta a gradini
    prev_q = 0
    for i, (q_i, price, up) in enumerate(zip(q_cum, p, up_labels)):
        ax.hlines(price, prev_q, q_i, colors='#43c07a' if market_type=='FCMT' else '#457b9d', linewidth=3)
        # Etichetta UP sull'ascissa in corrispondenza del salto di prezzo
        ax.text(q_i, price+3, up, ha='center', fontsize=10, color='black', rotation=45, fontweight='bold')
        prev_q = q_i
    # Domanda
    if len(p) > 0:
        ax.vlines(demand, min(p)-10, max(p)+10, colors='orange', linestyles='dashed', label='Domanda')
    # Prezzo marginale
    marginal_price = 0
    if acc:
        for a in acc:
            if a['q'] > 0:
                marginal_price = a['p']
    ax.axhline(marginal_price, color='purple', linestyle=':', label='Prezzo marginale')
    ax.set_xlabel('Quantità cumulata (MWh)')
    ax.set_ylabel('Prezzo (€/MWh)')
    ax.set_title(f'Curva di offerta e domanda - {market_type}')
    ax.legend()
    ax.grid(True)
    return fig

st.markdown("**FCMT**")
st.pyplot(plot_supply_demand_curve(offers, acc_fcmt, split[0], 'FCMT'))
st.markdown("**FCMNT**")
st.pyplot(plot_supply_demand_curve(offers, acc_fcmnt, split[1], 'FCMNT'))

# --- Curva domanda/offerta mercato unico (PaC classico) ---
def plot_supply_demand_classic(offers, accepted_classic, demand, marginal_price_classic):
    offers = sorted(offers, key=lambda x: x['p'])
    q = [o['q'] for o in offers]
    p = [o['p'] for o in offers]
    up_labels = [o['UP'] for o in offers]
    q_cum = np.cumsum(q)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    prev_q = 0
    for i, (q_i, price, up) in enumerate(zip(q_cum, p, up_labels)):
        ax.hlines(price, prev_q, q_i, colors='#888', linewidth=3)
        ax.text(q_i, price+3, up, ha='center', fontsize=10, color='black', rotation=45, fontweight='bold')
        prev_q = q_i
    ax.vlines(demand, min(p)-10, max(p)+10, colors='orange', linestyles='dashed', label='Domanda')
    ax.axhline(marginal_price_classic, color='purple', linestyle=':', label='Prezzo marginale')
    ax.set_xlabel('Quantità cumulata (MWh)')
    ax.set_ylabel('Prezzo (€/MWh)')
    ax.set_title('Curva di offerta e domanda nel Mercato Unico attuale (PaC classico)')
    ax.legend()
    ax.grid(True)
    return fig


# --- Visualizzazione UP accettate/escluse nel mercato unico ---
def plot_up_status_classic(offers, accepted_classic):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    fig, ax = plt.subplots(figsize=(8,1.7))
    # Tutte le UP su una riga
    for i, o in enumerate(offers):
        acc = next((a for a in accepted_classic if a['UP']==o['UP'] and a['q']>0), None)
        color = '#43c07a' if acc else '#e76f51'
        marker = 'o' if acc else 'x'
        text_color = 'black' if acc else 'black'
        ax.scatter(i, 0, s=400, color=color, marker=marker, edgecolor='k', zorder=3)
        # Etichetta spostata sopra il marker, con outline bianco spesso
        ax.text(i, 0.23, o['UP'], ha='center', fontsize=10, color=text_color, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.set_yticks([0])
    ax.set_yticklabels(['Mercato Unico'])
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(offers)-0.5)
    ax.set_ylim(-0.5, 0.8)
    ax.set_title('UP accettate (verde) ed escluse (rosso) - Mercato Unico')
    ax.axis('off')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Accettata', markerfacecolor='#43c07a', markersize=15, markeredgecolor='k'),
        Line2D([0], [0], marker='x', color='w', label='Esclusa', markerfacecolor='#e76f51', markeredgecolor='#e76f51', markersize=15)
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=10, frameon=False)
    return fig




# Usa solo la tabella interattiva (df) per tutte le visualizzazioni e simulazioni
# Conversione colonne per compatibilità con funzioni esistenti (già fatto sopra, riutilizza la variabile offers)
# Visualizzazione UP accettate/escluse nel mercato unico PRIMA della curva domanda/offerta
st.subheader("UP accettate ed escluse - Mercato Unico (PaC classico)")
st.pyplot(plot_up_status_classic(offers, accepted_classic))

st.subheader("Curva domanda/offerta - Mercato Unico (PaC classico)")
st.pyplot(plot_supply_demand_classic(offers, accepted_classic, demand, marginal_price_classic))

