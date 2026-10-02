#!/usr/bin/env python3
"""Authenticated frame-by-frame video encryption prototype.

The secure path intentionally uses standard cryptography. Chaos-based image
scrambling can be added as a research/visualization mode, but it should not be
used as the only protection for real video.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


MAGIC = b"VENC\x01"
HEADER_LENGTH = struct.Struct(">I")
RECORD_LENGTH = struct.Struct(">Q")
NONCE_PREFIX_BYTES = 4
SALT_BYTES = 16
KEY_BYTES = 32

# These parameters are deliberately explicit so the file format remains
# reproducible. N=2^15 is a reasonable local prototype setting; it can be
# increased for a production threat model after benchmarking.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1


class VideoEncryptionError(RuntimeError):
    """Raised when an encrypted container is invalid or cannot be processed."""


def _derive_key(password: str, salt: bytes) -> bytes:
    if not password:
        raise VideoEncryptionError("Password must not be empty")
    return Scrypt(
        salt=salt,
        length=KEY_BYTES,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
    ).derive(password.encode("utf-8"))


def _nonce(prefix: bytes, frame_index: int) -> bytes:
    if len(prefix) != NONCE_PREFIX_BYTES:
        raise VideoEncryptionError("Invalid nonce prefix")
    return prefix + frame_index.to_bytes(8, "big")


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise VideoEncryptionError("Encrypted file ended unexpectedly")
    return data


def _write_header(stream: BinaryIO, header: dict) -> bytes:
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    stream.write(MAGIC)
    stream.write(HEADER_LENGTH.pack(len(header_bytes)))
    stream.write(header_bytes)
    return header_bytes


def _read_header(stream: BinaryIO) -> tuple[dict, bytes]:
    if _read_exact(stream, len(MAGIC)) != MAGIC:
        raise VideoEncryptionError("Not a VENC v1 file")
    header_size = HEADER_LENGTH.unpack(_read_exact(stream, HEADER_LENGTH.size))[0]
    if header_size > 1_000_000:
        raise VideoEncryptionError("Unreasonable header size")
    header_bytes = _read_exact(stream, header_size)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VideoEncryptionError("Invalid VENC header") from exc
    required = {"version", "width", "height", "fps", "salt", "nonce_prefix"}
    if not required.issubset(header):
        raise VideoEncryptionError("VENC header is missing required fields")
    if header["version"] != 1:
        raise VideoEncryptionError(f"Unsupported VENC version: {header['version']}")
    return header, header_bytes


def _password_for(args: argparse.Namespace, confirm: bool = False) -> str:
    if args.password is not None:
        return args.password
    password = getpass.getpass("Password: ")
    if confirm:
        repeated = getpass.getpass("Repeat password: ")
        if password != repeated:
            raise VideoEncryptionError("Passwords do not match")
    return password


def encrypt_video(input_path: Path, output_path: Path, password: str) -> int:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise VideoEncryptionError(f"Could not open input video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not fps or fps != fps:  # NaN-safe fallback for unusual containers.
        fps = 30.0

    salt = os.urandom(SALT_BYTES)
    nonce_prefix = os.urandom(NONCE_PREFIX_BYTES)
    key = _derive_key(password, salt)
    cipher = AESGCM(key)

    frame_count = 0
    try:
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise VideoEncryptionError("Input video contains no readable frames")

        height, width = first_frame.shape[:2]
        header = {
            "version": 1,
            "algorithm": "AES-256-GCM",
            "kdf": "Scrypt",
            "scrypt": {"n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P},
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce_prefix": base64.b64encode(nonce_prefix).decode("ascii"),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_codec": "PNG",
            "frame_count": None,
            "source_name": input_path.name,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as stream:
            header_bytes = _write_header(stream, header)
            for frame in _frame_iterator(capture, first_frame):
                ok, encoded = cv2.imencode(
                    ".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]
                )
                if not ok:
                    raise VideoEncryptionError(
                        f"Could not PNG-encode frame {frame_count}"
                    )
                plaintext = encoded.tobytes()
                associated_data = header_bytes + frame_count.to_bytes(8, "big")
                ciphertext = cipher.encrypt(
                    _nonce(nonce_prefix, frame_count), plaintext, associated_data
                )
                stream.write(RECORD_LENGTH.pack(len(ciphertext)))
                stream.write(ciphertext)
                frame_count += 1
            stream.write(RECORD_LENGTH.pack(0))
    finally:
        capture.release()

    return frame_count


def _frame_iterator(capture: cv2.VideoCapture, first_frame):
    yield first_frame
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame


def decrypt_video(input_path: Path, output_path: Path, password: str) -> int:
    with input_path.open("rb") as stream:
        header, header_bytes = _read_header(stream)
        try:
            salt = base64.b64decode(header["salt"], validate=True)
            nonce_prefix = base64.b64decode(header["nonce_prefix"], validate=True)
        except (ValueError, TypeError) as exc:
            raise VideoEncryptionError("Invalid salt or nonce in VENC header") from exc

        key = _derive_key(password, salt)
        cipher = AESGCM(key)
        width = int(header["width"])
        height = int(header["height"])
        fps = float(header["fps"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            raise VideoEncryptionError(
                "Could not open FFV1 output. Use an OpenCV build with FFmpeg/FFV1 "
                "support or choose a compatible output path."
            )

        frame_count = 0
        try:
            while True:
                record_size = RECORD_LENGTH.unpack(
                    _read_exact(stream, RECORD_LENGTH.size)
                )[0]
                if record_size == 0:
                    break
                if record_size > 2**32:
                    raise VideoEncryptionError("Unreasonable encrypted frame size")
                ciphertext = _read_exact(stream, record_size)
                associated_data = header_bytes + frame_count.to_bytes(8, "big")
                try:
                    plaintext = cipher.decrypt(
                        _nonce(nonce_prefix, frame_count),
                        ciphertext,
                        associated_data,
                    )
                except Exception as exc:
                    raise VideoEncryptionError(
                        f"Authentication failed for frame {frame_count}; "
                        "password or file may be wrong/corrupted"
                    ) from exc

                decoded = cv2.imdecode(
                    np.frombuffer(plaintext, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if decoded is None or decoded.shape[:2] != (height, width):
                    raise VideoEncryptionError(
                        f"Decoded frame {frame_count} has unexpected dimensions"
                    )
                writer.write(decoded)
                frame_count += 1
        finally:
            writer.release()

    return frame_count


def show_info(input_path: Path) -> None:
    with input_path.open("rb") as stream:
        header, _ = _read_header(stream)
    printable = {
        key: value
        for key, value in header.items()
        if key not in {"salt", "nonce_prefix"}
    }
    printable["salt"] = "<hidden>"
    printable["nonce_prefix"] = "<hidden>"
    print(json.dumps(printable, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticated frame-by-frame video encryption prototype"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    encrypt = commands.add_parser("encrypt", help="Encrypt a video into .venc")
    encrypt.add_argument("input", type=Path)
    encrypt.add_argument("output", type=Path)
    encrypt.add_argument("--password", help="For demos; omit to prompt securely")

    decrypt = commands.add_parser("decrypt", help="Decrypt a .venc into a video")
    decrypt.add_argument("input", type=Path)
    decrypt.add_argument("output", type=Path)
    decrypt.add_argument("--password", help="For demos; omit to prompt securely")

    info = commands.add_parser("info", help="Show non-secret container metadata")
    info.add_argument("input", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "encrypt":
            frames = encrypt_video(
                args.input, args.output, _password_for(args, confirm=True)
            )
            print(f"Encrypted {frames} frames -> {args.output}")
        elif args.command == "decrypt":
            frames = decrypt_video(
                args.input, args.output, _password_for(args, confirm=False)
            )
            print(f"Decrypted {frames} frames -> {args.output}")
        else:
            show_info(args.input)
        return 0
    except (OSError, VideoEncryptionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
