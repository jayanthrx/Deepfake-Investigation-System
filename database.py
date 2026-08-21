from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Investigation(db.Model):
    __tablename__ = "investigation"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    filename = db.Column(
        db.String(200),
        nullable=False
    )

    result = db.Column(
        db.String(50),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=False
    )

    risk = db.Column(
        db.String(50),
        nullable=False
    )

    date = db.Column(
        db.String(100),
        nullable=False
    )

    def __repr__(self):
        return f"<Investigation {self.filename}>"