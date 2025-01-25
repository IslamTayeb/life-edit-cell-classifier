import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_and_process_data(counts_file, metadata_file):
    df = pd.read_csv(counts_file, sep="\t")
    df = df.set_index("GeneID")

    # Log transform the counts
    df = np.log2(df + 1)

    # Z-score standardization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.T)
    scaled_df = pd.DataFrame(scaled_data)
    scaled_df.columns = df.index
    scaled_df.index = df.columns

    metadata = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    return scaled_df, metadata

def filter_genes(
    scaled_df,
    df_ut,
    min_median_difference=1.6,
    min_standard_deviation_percent_difference=250,
):
    relevant_genes = []

    for i in range(len(scaled_df.columns)):
        col_data_edited = scaled_df.iloc[:, i]
        col_data_ut = df_ut.iloc[:, i]

        median_edited = col_data_edited.median()
        median_ut = col_data_ut.median()
        std_edited = col_data_edited.std()
        std_ut = col_data_ut.std()

        if (
            abs(median_edited - median_ut) >= min_median_difference
            or (
                std_ut != 0
                and std_edited / std_ut
                >= (1 + min_standard_deviation_percent_difference / 100)
            )
            or (
                std_edited != 0
                and std_ut / std_edited
                >= (1 + min_standard_deviation_percent_difference / 100)
            )
        ):
            relevant_genes.append(scaled_df.columns[i])

    return relevant_genes


def merge_with_metadata(filtered_df, metadata):
    transposed_data = filtered_df.transpose().reset_index()
    transposed_data.columns.values[0] = "GeneID"
    transposed_data["GeneID"] = transposed_data["GeneID"].astype(int)
    merged_data = transposed_data.merge(metadata, on="GeneID", how="left").set_index("GeneID")
    return merged_data

def perform_pca_analysis(filtered_df):
    expression_cols = filtered_df.columns[1:39]  # Get expression columns
    expression_data = filtered_df[expression_cols].apply(pd.to_numeric, errors='coerce')

    pca = PCA()
    pca_result = pca.fit_transform(expression_data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    return pca, pca_result, cumulative_variance

def plot_variance_explained(pca):
    fig = go.Figure()

    x_values = list(range(1, len(pca.explained_variance_ratio_) + 1))

    fig.add_trace(
        go.Bar(x=x_values, y=pca.explained_variance_ratio_, name="Individual")
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=np.cumsum(pca.explained_variance_ratio_),
            name="Cumulative",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title="Explained Variance Ratio by Principal Component",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
        showlegend=True,
        height=500,
    )

    return fig


def analyze_pca_components(pca_result, n_components, sample_names):
    pca_df = pd.DataFrame(
        pca_result[:, :n_components],
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=sample_names
    )

    # Create correlation heatmap between PCs
    corr_matrix = pca_df.corr()

    # Remove diagonal entries
    np.fill_diagonal(corr_matrix.values, np.nan)

    # Get max absolute correlation for symmetric scale
    max_corr = max(abs(corr_matrix.min().min()), abs(corr_matrix.max().max()))

    fig = px.imshow(
        corr_matrix,
        title="PCA Components Correlation Heatmap",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",
        zmin=-max_corr,
        zmax=max_corr,
        color_continuous_midpoint=0,
        aspect="equal"
    )

    # fig.update_traces(
    #     text=corr_matrix.round(3),
    #     texttemplate="%{text:.3f}",
    #     textfont={"size": 12}
    # )

    return pca_df, fig

def get_relevant_genes_with_contribution(pca, filtered_df, pc_idx, cumulative_threshold=0.9):
    # Get absolute loadings
    loadings = pd.DataFrame(
        abs(pca.components_[pc_idx]),
        index=filtered_df.columns[:38],
        columns=["Loading"]
    ).sort_values("Loading", ascending=False)

    cumulative_variance = loadings["Loading"].cumsum() / loadings["Loading"].sum()
    relevant_genes = loadings[cumulative_variance <= cumulative_threshold]
    contribution_percentages = (relevant_genes["Loading"] / loadings["Loading"].sum()) * 100

    return pd.DataFrame({
        "Loading": relevant_genes["Loading"],
        "Contribution (%)": contribution_percentages
    })


def main():
    st.title("Gene Expression Analysis Dashboard")

    st.header("Data Upload")
    counts_file = st.file_uploader("Upload raw counts file (TSV format)", type=["tsv"])
    metadata_file = st.file_uploader("Upload metadata file (TSV format)", type=["tsv"])

    if counts_file and metadata_file:
        try:
            scaled_df, metadata = load_and_process_data(counts_file, metadata_file)

            st.header("Data Overview")
            st.write(f"Number of samples: {scaled_df.shape[0]}")
            st.write(f"Number of genes: {scaled_df.shape[1]}")
            st.dataframe(scaled_df.head())

            mechanisms = {
                "BE4": [
                    "GSM6745599",
                    "GSM6745600",
                    "GSM6745601",
                    "GSM6745611",
                    "GSM6745612",
                    "GSM6745613",
                ],
                "ABE8": [
                    "GSM6745602",
                    "GSM6745603",
                    "GSM6745604",
                    "GSM6745614",
                    "GSM6745615",
                    "GSM6745616",
                ],
                "Cas9": [
                    "GSM6745605",
                    "GSM6745606",
                    "GSM6745607",
                    "GSM6745617",
                    "GSM6745618",
                    "GSM6745619",
                ],
                "Utelectro": [
                    "GSM6745609",
                    "GSM6745610",
                    "GSM6745620",
                    "GSM6745621",
                    "GSM6745622",
                ],
                "dCas9": ["GSM6745623", "GSM6745624", "GSM6745625"],
                "BE4alone": ["GSM6745626", "GSM6745627", "GSM6745628"],
                "ABE8alone": ["GSM6745629", "GSM6745630", "GSM6745631"],
                "UT": [
                    "GSM6745632",
                    "GSM6745633",
                    "GSM6745634",
                    "GSM6745635",
                    "GSM6745636",
                    "GSM6745637",
                ],
            }

            df_edited = scaled_df.loc[
                sum([mechanisms[k] for k in mechanisms.keys() if k != "UT"], [])
            ]
            df_ut = scaled_df.loc[mechanisms["UT"]]

            st.header("Gene Filtering Parameters")
            median_diff = st.slider("Minimum median difference", 0.0, 5.0, 1.6, 0.1)
            std_diff = st.slider(
                "Minimum standard deviation percent difference", 0, 500, 250, 25
            )

            if st.button("Apply Filtering"):
                relevant_genes = filter_genes(df_edited, df_ut, median_diff, std_diff)
                filtered_df = scaled_df[relevant_genes]
                st.write(f"Number of genes after filtering: {len(relevant_genes)}")
                st.write("Filtered DataFrame:")
                st.dataframe(filtered_df)
                st.session_state["filtered_df"] = filtered_df
                st.session_state["pca_performed"] = False

            if 'filtered_df' in st.session_state:
                filtered_df = st.session_state['filtered_df']
                pca, pca_result, cumulative_variance = perform_pca_analysis(filtered_df)

                pca_tab, genes_tab = st.tabs(["PCA Analysis", "Gene Contributions"])

                with pca_tab:
                    st.subheader("PCA Variance Analysis")
                    variance_fig = plot_variance_explained(pca)
                    st.plotly_chart(variance_fig, key="variance_plot")

                    desired_variance = st.slider(
                        "Select desired explained variance ratio (0-1)",
                        min_value=0.0, max_value=1.0, value=0.75, step=0.05
                    )

                    n_components = np.argmax(cumulative_variance >= desired_variance) + 1
                    st.write(f"Number of components needed to explain {desired_variance*100:.1f}% of variance: {n_components}")

                    st.subheader("PCA Components Analysis")
                    pca_df, corr_fig = analyze_pca_components(pca_result, min(n_components, 30), filtered_df.index)
                    st.plotly_chart(corr_fig, key="correlation_plot")
                    st.dataframe(pca_df.describe())

                    if n_components >= 2:
                        scatter_fig = px.scatter(
                            pca_df, x="PC1", y="PC2",
                            hover_name=pca_df.index,
                            title="PC1 vs PC2 Scatter Plot"
                        )
                        st.plotly_chart(scatter_fig, key="scatter_2d")

                        if n_components >= 3:
                            scatter_3d = px.scatter_3d(
                                pca_df, x="PC1", y="PC2", z="PC3",
                                hover_name=pca_df.index,
                                title="First 3 Principal Components"
                            )
                            st.plotly_chart(scatter_3d, key="scatter_3d")

                with genes_tab:
                    st.subheader("Gene Contributions Analysis")
                    n_pcs = len(pca.components_)
                    pcs_per_tab = 5
                    n_tabs = (n_pcs + pcs_per_tab - 1) // pcs_per_tab

                    tab_titles = [f"PCs {i*pcs_per_tab+1}-{min((i+1)*pcs_per_tab, n_pcs)}" for i in range(n_tabs)]
                    pc_tabs = st.tabs(tab_titles)

                    for tab_idx, tab in enumerate(pc_tabs):
                        with tab:
                            start_pc = tab_idx * pcs_per_tab
                            end_pc = min((tab_idx + 1) * pcs_per_tab, n_pcs)

                            for pc_idx in range(start_pc, end_pc):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**PC{pc_idx + 1} - Top Contributing Genes:**")
                                    contributions = get_relevant_genes_with_contribution(pca, filtered_df, pc_idx)
                                    st.dataframe(contributions.head(10), key=f"pc_{pc_idx+1}_contributions")

                                if pc_idx + 1 < end_pc:
                                    with col2:
                                        st.write(f"**PC{pc_idx + 2} - Top Contributing Genes:**")
                                        contributions = get_relevant_genes_with_contribution(pca, filtered_df, pc_idx + 1)
                                        st.dataframe(contributions.head(10), key=f"pc_{pc_idx+2}_contributions")

                                st.markdown("---")
                if not st.session_state.get("pca_performed", False):
                    pca, pca_result, cumulative_variance = perform_pca_analysis(
                        filtered_df
                    )
                    st.session_state["pca"] = pca
                    st.session_state["pca_result"] = pca_result
                    st.session_state["cumulative_variance"] = cumulative_variance
                    st.session_state["pca_performed"] = True

                pca = st.session_state["pca"]
                pca_result = st.session_state["pca_result"]
                cumulative_variance = st.session_state["cumulative_variance"]

                desired_variance = st.slider(
                    "Select desired explained variance ratio (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.05,
                )

                n_components = np.argmax(cumulative_variance >= desired_variance) + 1
                st.write(
                    f"Number of components needed to explain {desired_variance*100:.1f}% of variance: {n_components}"
                )

                st.subheader("PCA Components Analysis")
                pca_df, corr_fig = analyze_pca_components(
                    pca_result, min(n_components, 30), filtered_df.index
                )
                st.plotly_chart(corr_fig)
                st.dataframe(pca_df.describe())

                if n_components >= 2:
                    scatter_fig = px.scatter(
                        pca_df,
                        x="PC1",
                        y="PC2",
                        hover_name=pca_df.index,
                        title="PC1 vs PC2 Scatter Plot",
                    )
                    st.plotly_chart(scatter_fig)

                    if n_components >= 3:
                        scatter_3d = px.scatter_3d(
                            pca_df,
                            x="PC1",
                            y="PC2",
                            z="PC3",
                            hover_name=pca_df.index,
                            title="First 3 Principal Components",
                        )
                        st.plotly_chart(scatter_3d)

                st.subheader("Top Gene Contributions to Principal Components")
                for i in range(min(n_components, 5)):
                    loadings = pd.DataFrame(
                        pca.components_[i],
                        columns=["Loading"],
                        index=filtered_df.columns,
                    ).sort_values("Loading", ascending=False)
                    st.write(f"PC{i+1} - Top 10 genes by absolute loading:")
                    st.dataframe(loadings.head(10))

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")


if __name__ == "__main__":
    main()
