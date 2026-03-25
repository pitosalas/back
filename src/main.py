import marimo

app = marimo.App()

@app.cell
def _():
    print("hello from back")
    return ()

if __name__ == "__main__":
    app.run()
