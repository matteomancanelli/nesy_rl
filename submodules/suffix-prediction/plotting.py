import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_loss_over_epoch(values, title, folder):
    plt.figure(figsize=(10, 5))
    plt.plot(values)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{folder}/{title}.png", dpi=300, bbox_inches='tight')
    plt.close()


custom_palette = {
    'baseline | greedy': '#35c000',  # light green
    'with BK | greedy': '#175300',  # dark green
    'baseline | temperature': '#ef0000',  # light red
    'with BK | temperature': '#8b0000',  # dark red
}

model_order = [
    'baseline | greedy',
    'with BK | greedy',
    'baseline | temperature',
    'with BK | temperature',
]


def plot_metrics_as_bars(data, metric_name, prefixes, path, errorbar=False):
    prefix_order = [
        f'train | {prefixes[0]}',
        f'train | {prefixes[1]}',
        f'train | {prefixes[2]}',
        f'test | {prefixes[0]}',
        f'test | {prefixes[1]}',
        f'test | {prefixes[2]}',
    ]

    melted = data.melt(
        id_vars=['prefix length', 'model', 'sampling strategy'],
        value_vars=['train accuracy', 'test accuracy', 'train DL', 'test DL', 'train DL scaled', 'test DL scaled', 'train sat', 'test sat'],
        var_name='metric_type',
        value_name='value'
    )

    melted['split'] = melted['metric_type'].apply(lambda x: x.split()[0])
    melted['metric'] = melted['metric_type'].apply(lambda x: ' '.join(x.split()[1:]))

    melted['prefix_split'] = melted['split'] + ' | ' + melted['prefix length'].astype(str)
    melted['model_sampling'] = melted['model'] + ' | ' + melted['sampling strategy']
    data = melted[melted['metric'] == metric_name]

    grouped = data.groupby(['prefix_split', 'model_sampling'])['value']
    stats = grouped.agg(['min', 'mean', 'max']).reset_index()

    stats['prefix_split'] = pd.Categorical(stats['prefix_split'], categories=prefix_order, ordered=True)
    stats['model_sampling'] = pd.Categorical(stats['model_sampling'], categories=model_order, ordered=True)
    ordered_pairs = [(model, prefix) for model in model_order for prefix in prefix_order]
    stats = stats.set_index(['model_sampling', 'prefix_split']).loc[ordered_pairs].reset_index()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=stats, x='prefix_split', y='mean', hue='model_sampling', hue_order=model_order, palette=custom_palette)
    ax.grid(axis='y', color='black', linestyle='--')
    for bar, (_, row) in zip(ax.patches, stats.iterrows()):
        if errorbar:
            x = bar.get_x() + bar.get_width() / 2
            #print(f'Metric: {metric_name}; Min: {row['min']}; Mean: {row['mean']}; Max: {row['max']}')
            error_low = max(0, row['mean'] - row['min'])
            error_high = max(0, row['max'] - row['mean'])

            ax.errorbar(x, row['mean'], yerr=[[error_low], [error_high]], ecolor='black', capsize=3)
        bar.set_edgecolor('black')
        bar.set_linewidth(1)

    plt.title(f'{metric_name.capitalize()} by Split and Prefix')
    plt.xlabel('Split | Prefix Length')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=len(data['model_sampling'].unique()))

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
