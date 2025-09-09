import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # MCMC methods for LLC

    In which we ground truth some useful samplers

    """
    )
    return


@app.function
def example(args):
    "test function"
    return f"I am an example function: {args}"


@app.cell
def _():
    print(example("foo"))
    return


if __name__ == "__main__":
    app.run()
