# Generated by Django 3.1.3 on 2020-12-10 22:10

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    replaces = [
        ("dcodex_collation", "0001_initial"),
        ("dcodex_collation", "0002_auto_20200822_2207"),
        ("dcodex_collation", "0003_remove_alignment_matrix_data"),
        ("dcodex_collation", "0004_atext_transition_transitiontype"),
        ("dcodex_collation", "0005_auto_20200825_1717"),
        ("dcodex_collation", "0006_auto_20200825_2146"),
        ("dcodex_collation", "0007_auto_20200825_2216"),
        ("dcodex_collation", "0008_cell_state_token"),
        ("dcodex_collation", "0009_token_rank"),
        ("dcodex_collation", "0010_auto_20200826_0856"),
        ("dcodex_collation", "0011_auto_20200826_0857"),
        ("dcodex_collation", "0012_auto_20200826_0907"),
        ("dcodex_collation", "0013_auto_20200826_1014"),
        ("dcodex_collation", "0014_auto_20200826_2202"),
        ("dcodex_collation", "0015_transition"),
        ("dcodex_collation", "0016_remove_state_column"),
        ("dcodex_collation", "0017_auto_20200829_1339"),
        ("dcodex_collation", "0018_rate_ratesystem_transitionrate"),
        ("dcodex_collation", "0019_auto_20201102_1048"),
        ("dcodex_collation", "0020_auto_20201116_2146"),
        ("dcodex_collation", "0021_remove_row_tokens"),
    ]

    initial = True

    dependencies = [
        ("dcodex", "0025_auto_20200809_1536"),
    ]

    operations = [
        migrations.CreateModel(
            name="Alignment",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "family",
                    models.ForeignKey(
                        blank=True,
                        default=None,
                        null=True,
                        on_delete=django.db.models.deletion.SET_DEFAULT,
                        to="dcodex.family",
                    ),
                ),
                (
                    "verse",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="dcodex.verse"
                    ),
                ),
            ],
            options={
                "ordering": ["family", "verse"],
            },
        ),
        migrations.CreateModel(
            name="Column",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "order",
                    models.PositiveIntegerField(
                        verbose_name="The rank of this column in the alignment"
                    ),
                ),
                (
                    "alignment",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.alignment",
                    ),
                ),
            ],
            options={
                "ordering": ["order"],
            },
        ),
        migrations.CreateModel(
            name="Row",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "alignment",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.alignment",
                    ),
                ),
                (
                    "transcription",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex.versetranscription",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="TransitionType",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=255)),
                (
                    "inverse_name",
                    models.CharField(
                        blank=True, default=None, max_length=255, null=True
                    ),
                ),
            ],
            options={
                "ordering": ["name"],
            },
        ),
        migrations.CreateModel(
            name="Token",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "text",
                    models.CharField(
                        help_text="The characters of this token/word as they appear in the manuscript text.",
                        max_length=255,
                    ),
                ),
                (
                    "regularized",
                    models.CharField(
                        help_text="A regularized form of the text of this token.",
                        max_length=255,
                    ),
                ),
                (
                    "alignment",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.alignment",
                    ),
                ),
                ("rank", models.PositiveIntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name="State",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "text",
                    models.CharField(
                        blank=True,
                        help_text="A regularized form for the text of this state.",
                        max_length=255,
                        null=True,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Transition",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("inverse", models.BooleanField()),
                (
                    "column",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.column",
                    ),
                ),
                (
                    "end_state",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="end_state",
                        to="dcodex_collation.state",
                    ),
                ),
                (
                    "start_state",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="start_state",
                        to="dcodex_collation.state",
                    ),
                ),
                (
                    "transition_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.transitiontype",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Cell",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "column",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.column",
                    ),
                ),
                (
                    "row",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.row",
                    ),
                ),
                (
                    "state",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.state",
                    ),
                ),
                (
                    "token",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.token",
                    ),
                ),
            ],
            options={
                "ordering": ["column", "row"],
            },
        ),
        migrations.AddField(
            model_name="column",
            name="atext",
            field=models.ForeignKey(
                blank=True,
                default=None,
                null=True,
                on_delete=django.db.models.deletion.SET_DEFAULT,
                to="dcodex_collation.state",
            ),
        ),
        migrations.AddField(
            model_name="column",
            name="atext_notes",
            field=models.TextField(blank=True, default=None, null=True),
        ),
        migrations.CreateModel(
            name="Rate",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name="RateSystem",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=255)),
                (
                    "default_rate",
                    models.ForeignKey(
                        blank=True,
                        default=None,
                        null=True,
                        on_delete=django.db.models.deletion.SET_DEFAULT,
                        to="dcodex_collation.rate",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="TransitionRate",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "inverse",
                    models.BooleanField(
                        blank=True,
                        default=None,
                        help_text="If None, then the rate applies in both directions.",
                        null=True,
                    ),
                ),
                (
                    "rate",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.rate",
                    ),
                ),
                (
                    "system",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.ratesystem",
                    ),
                ),
                (
                    "transition_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="dcodex_collation.transitiontype",
                    ),
                ),
            ],
        ),
    ]
