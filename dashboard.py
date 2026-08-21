@app.route("/dashboard")
def dashboard():

    records = Investigation.query.order_by(
        Investigation.id.desc()
    ).all()

    total = Investigation.query.count()

    fake = Investigation.query.filter_by(
        result="Fake"
    ).count()

    real = Investigation.query.filter_by(
        result="Real"
    ).count()

    uncertain = Investigation.query.filter_by(
        result="Uncertain"
    ).count()

    if total > 0:

        avg_confidence = round(
            sum(r.confidence for r in records) / total,
            2
        )

    else:

        avg_confidence = 0

    return render_template(
        "dashboard.html",
        records=records,
        total=total,
        fake=fake,
        real=real,
        uncertain=uncertain,
        avg_confidence=avg_confidence
    )