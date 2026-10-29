from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="DataDoctor: Multi-View Label-Error Detection & Hygiene CLI")
console = Console()


@app.command()
def info():
    """Prints DataDoctor library information and environment status."""
    console.print("[bold green]DataDoctor[/bold green] v2.1.0")
    console.print("Agent-Ready Multi-View Label-Error Detection, Hygiene & Disagreement-Routed Repair")


@app.command()
def audit(
    dataset: str = typer.Argument("ag_news", help="Dataset name ('ag_news', 'cifar10') or path to CSV"),
    subsample: Optional[int] = typer.Option(500, help="Subsample train size for fast audit"),
    output: str = typer.Option("scores.parquet", help="Path to export scores dataframe"),
):
    """Runs end-to-end multi-view audit and outputs ranked suspicion scores."""
    console.print(f"[bold blue]Starting audit for {dataset}...[/bold blue]")
    from datadoctor.io.loaders import load_ag_news, load_from_csv
    from datadoctor.extractors.text import extract_text_embeddings
    from datadoctor.detectors.oof_probe import compute_oof_probabilities
    from datadoctor.detectors.v1_self_conf import compute_v1_scores
    from datadoctor.detectors.v2_knn import compute_v2_scores
    from datadoctor.detectors.v3_cross_probe import compute_v3_scores
    from datadoctor.fusion.rank_fusion import compute_rank_fusion
    from datadoctor.router.rules import assign_route_buckets
    from datadoctor.hygiene.scanner import scan_dataset_hygiene

    if dataset == "ag_news":
        artifact = load_ag_news(subsample_train=subsample)
    elif Path(dataset).exists():
        artifact = load_from_csv(dataset, text_col="text", label_col="label")
    else:
        console.print(f"[bold red]Unknown dataset: {dataset}[/bold red]")
        raise typer.Exit(code=1)

    # Feature extraction
    console.print("Extracting Backbone A & B features...")
    artifact.embeddings["A"] = extract_text_embeddings(artifact.raw_references, "minilm_l6")
    artifact.embeddings["B"] = extract_text_embeddings(artifact.raw_references, "bge_small")

    # OOF Probes
    console.print("Fitting OOF Probes on train split...")
    t_mask = artifact.train_mask
    artifact.oof_probs["A"] = compute_oof_probabilities(
        artifact.embeddings["A"][t_mask], artifact.given_labels[t_mask], len(artifact.class_names)
    )
    artifact.oof_probs["B"] = compute_oof_probabilities(
        artifact.embeddings["B"][t_mask], artifact.given_labels[t_mask], len(artifact.class_names)
    )

    # Views
    artifact.view_scores["v1"] = compute_v1_scores(artifact)
    artifact.view_scores["v2"] = compute_v2_scores(artifact, "A")
    artifact.view_scores["v3"] = compute_v3_scores(artifact)

    # Fusion & Routing
    compute_rank_fusion(artifact)
    assign_route_buckets(artifact)

    # Export
    df = artifact.to_scores_dataframe()
    df.to_parquet(output)
    console.print(f"[bold green]Saved {len(df)} ranked sample scores to {output}[/bold green]")


@app.command()
def hygiene(
    dataset: str = typer.Argument("ag_news", help="Dataset name ('ag_news', 'cifar10') or path to CSV"),
    subsample: Optional[int] = typer.Option(500, help="Subsample train size"),
):
    """Executes 3-pass hygiene scanner for exact duplicates, near-duplicates, and leakage."""
    from datadoctor.io.loaders import load_ag_news
    from datadoctor.extractors.text import extract_text_embeddings
    from datadoctor.hygiene.scanner import scan_dataset_hygiene

    if dataset == "ag_news":
        artifact = load_ag_news(subsample_train=subsample)
    else:
        console.print(f"[bold red]Dataset {dataset} not supported for quick hygiene demo[/bold red]")
        raise typer.Exit(code=1)

    artifact.embeddings["A"] = extract_text_embeddings(artifact.raw_references, "minilm_l6")
    report = scan_dataset_hygiene(artifact)

    table = Table(title=f"Hygiene Report: {dataset}")
    table.add_column("Hygiene Metric", style="cyan")
    table.add_column("Count", style="magenta")

    for k, v in report.summary.items():
        table.add_row(k, str(v))

    console.print(table)


if __name__ == "__main__":
    app()

