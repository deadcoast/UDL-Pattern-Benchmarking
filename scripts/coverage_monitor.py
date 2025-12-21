#!/usr/bin/env python3
"""
Automated Coverage Monitoring Script

This script provides automated coverage monitoring and alerting for the UDL Rating Framework.
It can be run in CI/CD pipelines or as a scheduled task to track coverage trends and
send alerts when coverage drops below acceptable thresholds.
"""

import os
import sys
import json
import sqlite3
import datetime
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@dataclass
class CoverageData:
    """Coverage data for a module or the entire project."""

    name: str
    statements: int
    missing: int
    coverage: float
    missing_lines: List[str]
    timestamp: datetime.datetime


@dataclass
class CoverageReport:
    """Complete coverage report."""

    overall: CoverageData
    modules: List[CoverageData]
    timestamp: datetime.datetime
    test_results: Dict[str, int]


class CoverageMonitor:
    """Main coverage monitoring class."""

    def __init__(
        self,
        db_path: str = "coverage_history.db",
        threshold: float = 90.0,
        critical_threshold: float = 85.0,
    ):
        self.db_path = Path(db_path)
        self.threshold = threshold
        self.critical_threshold = critical_threshold
        self.init_database()

    def init_database(self):
        """Initialize the coverage history database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coverage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_coverage REAL NOT NULL,
                total_statements INTEGER NOT NULL,
                missing_statements INTEGER NOT NULL,
                modules_data TEXT NOT NULL,
                test_results TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coverage_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                coverage_before REAL,
                coverage_after REAL,
                resolved BOOLEAN DEFAULT FALSE
            )
        """
        )
        conn.commit()
        conn.close()

    def run_tests_with_coverage(self) -> Tuple[bool, Dict[str, int]]:
        """Run tests with coverage collection."""
        print("Running tests with coverage...")

        # Run tests with coverage
        result = subprocess.run(
            [
                "uv",
                "run",
                "coverage",
                "run",
                "--source=udl_rating_framework",
                "-m",
                "pytest",
                "--tb=no",
                "-q",
            ],
            capture_output=True,
            text=True,
        )

        # Parse test results
        test_results = self._parse_test_results(result.stdout)

        return result.returncode == 0, test_results

    def _parse_test_results(self, output: str) -> Dict[str, int]:
        """Parse pytest output to extract test counts."""
        results = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        # Look for summary line like "105 failed, 488 passed, 5 skipped"
        lines = output.split("\n")
        for line in lines:
            if ("failed" in line or "passed" in line) and ("=" not in line):
                # Split by comma and parse each part
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    words = part.split()
                    if len(words) >= 2:
                        try:
                            count = int(words[0])
                            if "failed" in part:
                                results["failed"] = count
                            elif "passed" in part:
                                results["passed"] = count
                            elif "skipped" in part:
                                results["skipped"] = count
                            elif "error" in part:
                                results["errors"] = count
                        except ValueError:
                            continue
                break

        return results

    def _parse_coverage_text(self, output: str) -> float:
        """Parse coverage percentage from text output."""
        lines = output.split("\n")
        for line in lines:
            if "TOTAL" in line and "%" in line:
                parts = line.split()
                for part in parts:
                    if part.endswith("%"):
                        try:
                            return float(part[:-1])
                        except ValueError:
                            continue
        return 0.0

    def _parse_json_coverage(self, coverage_json: dict) -> CoverageReport:
        """Parse coverage data from JSON format."""
        # Parse overall coverage
        totals = coverage_json["totals"]
        overall = CoverageData(
            name="TOTAL",
            statements=totals["num_statements"],
            missing=totals["missing_lines"],
            coverage=totals["percent_covered"],
            missing_lines=[],
            timestamp=datetime.datetime.now(),
        )

        # Parse module coverage
        modules = []
        for filename, data in coverage_json["files"].items():
            # Convert absolute path to module name
            module_name = filename.replace("/", ".").replace(".py", "")
            if module_name.startswith("udl_rating_framework."):
                modules.append(
                    CoverageData(
                        name=module_name,
                        statements=data["summary"]["num_statements"],
                        missing=data["summary"]["missing_lines"],
                        coverage=data["summary"]["percent_covered"],
                        missing_lines=data.get("missing_lines", []),
                        timestamp=datetime.datetime.now(),
                    )
                )

        return CoverageReport(
            overall=overall,
            modules=modules,
            timestamp=datetime.datetime.now(),
            test_results={"passed": 0, "failed": 0, "skipped": 0, "errors": 0},
        )

    def _parse_text_coverage(self, output: str) -> CoverageReport:
        """Parse coverage data from text format."""
        lines = output.split("\n")
        modules = []
        overall_coverage = 0.0
        total_statements = 0
        total_missing = 0

        for line in lines:
            if (
                line.strip()
                and not line.startswith("-")
                and not line.startswith("Name")
            ):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        name = parts[0]
                        statements = int(parts[1])
                        missing = int(parts[2])
                        coverage = (
                            float(parts[3][:-1]) if parts[3].endswith("%") else 0.0
                        )

                        if name == "TOTAL":
                            overall_coverage = coverage
                            total_statements = statements
                            total_missing = missing
                        elif name.startswith("udl_rating_framework"):
                            modules.append(
                                CoverageData(
                                    name=name,
                                    statements=statements,
                                    missing=missing,
                                    coverage=coverage,
                                    missing_lines=[],
                                    timestamp=datetime.datetime.now(),
                                )
                            )
                    except (ValueError, IndexError):
                        continue

        overall = CoverageData(
            name="TOTAL",
            statements=total_statements,
            missing=total_missing,
            coverage=overall_coverage,
            missing_lines=[],
            timestamp=datetime.datetime.now(),
        )

        return CoverageReport(
            overall=overall,
            modules=modules,
            timestamp=datetime.datetime.now(),
            test_results={"passed": 0, "failed": 0, "skipped": 0, "errors": 0},
        )

    def collect_coverage_data(self) -> CoverageReport:
        """Collect current coverage data."""
        print("Collecting coverage data...")

        # Generate coverage report in text format first to get basic stats
        result = subprocess.run(
            ["uv", "run", "coverage", "report"], capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Coverage report failed: {result.stderr}")

        # Try to get JSON format for detailed data
        json_result = subprocess.run(
            ["uv", "run", "coverage", "json"], capture_output=True, text=True
        )

        if json_result.returncode == 0:
            try:
                coverage_json = json.loads(json_result.stdout)
                return self._parse_json_coverage(coverage_json)
            except json.JSONDecodeError:
                pass

        # Fallback to text parsing
        return self._parse_text_coverage(result.stdout)

    def store_coverage_data(self, report: CoverageReport):
        """Store coverage data in the database."""
        conn = sqlite3.connect(self.db_path)

        # Convert modules to serializable format
        modules_data = []
        for m in report.modules:
            module_dict = asdict(m)
            module_dict["timestamp"] = m.timestamp.isoformat()
            modules_data.append(module_dict)

        conn.execute(
            """
            INSERT INTO coverage_history 
            (timestamp, overall_coverage, total_statements, missing_statements, modules_data, test_results)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                report.timestamp.isoformat(),
                report.overall.coverage,
                report.overall.statements,
                report.overall.missing,
                json.dumps(modules_data),
                json.dumps(report.test_results),
            ),
        )

        conn.commit()
        conn.close()
        print(f"Stored coverage data: {report.overall.coverage:.1f}%")

    def check_coverage_thresholds(self, report: CoverageReport) -> List[str]:
        """Check coverage against thresholds and return alerts."""
        alerts = []

        # Check overall coverage
        if report.overall.coverage < self.critical_threshold:
            alerts.append(
                f"🚨 CRITICAL: Overall coverage {report.overall.coverage:.1f}% is below critical threshold {self.critical_threshold}%"
            )
        elif report.overall.coverage < self.threshold:
            alerts.append(
                f"⚠️ WARNING: Overall coverage {report.overall.coverage:.1f}% is below target threshold {self.threshold}%"
            )

        # Check module-specific thresholds
        critical_modules = [
            "udl_rating_framework.core",
            "udl_rating_framework.evaluation",
            "udl_rating_framework.models",
        ]

        for module in report.modules:
            if any(module.name.startswith(critical) for critical in critical_modules):
                if module.coverage < 95.0:
                    alerts.append(
                        f"⚠️ Critical module {module.name} has {module.coverage:.1f}% coverage (target: 95%)"
                    )

        # Check for significant drops
        previous_coverage = self.get_previous_coverage()
        if previous_coverage and report.overall.coverage < previous_coverage - 5.0:
            alerts.append(
                f"📉 Significant coverage drop: {previous_coverage:.1f}% → {report.overall.coverage:.1f}%"
            )

        return alerts

    def get_previous_coverage(self) -> Optional[float]:
        """Get the most recent coverage percentage."""
        conn = sqlite3.connect(self.db_path)
        result = conn.execute(
            """
            SELECT overall_coverage FROM coverage_history 
            ORDER BY timestamp DESC LIMIT 1
        """
        ).fetchone()
        conn.close()

        return result[0] if result else None

    def generate_coverage_report(self, report: CoverageReport) -> str:
        """Generate a formatted coverage report."""
        lines = [
            "# Coverage Report",
            f"**Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Overall Coverage:** {report.overall.coverage:.1f}%",
            f"- **Total Statements:** {report.overall.statements:,}",
            f"- **Missing Statements:** {report.overall.missing:,}",
            f"- **Tests Passed:** {report.test_results.get('passed', 0)}",
            f"- **Tests Failed:** {report.test_results.get('failed', 0)}",
            f"- **Tests Skipped:** {report.test_results.get('skipped', 0)}",
            "",
            "## Module Coverage",
            "",
        ]

        # Sort modules by coverage (lowest first)
        sorted_modules = sorted(report.modules, key=lambda m: m.coverage)

        # Add modules below threshold
        below_threshold = [m for m in sorted_modules if m.coverage < self.threshold]
        if below_threshold:
            lines.extend([f"### Modules Below {self.threshold}% Threshold", ""])
            for module in below_threshold:
                lines.append(
                    f"- **{module.name}:** {module.coverage:.1f}% ({module.missing} missing)"
                )
            lines.append("")

        # Add top performers
        top_performers = [m for m in sorted_modules if m.coverage >= 95.0][-10:]
        if top_performers:
            lines.extend(["### Top Performing Modules", ""])
            for module in reversed(top_performers):
                lines.append(f"- **{module.name}:** {module.coverage:.1f}%")
            lines.append("")

        return "\n".join(lines)

    def send_alert_email(self, alerts: List[str], report: CoverageReport):
        """Send coverage alerts via email."""
        if not alerts:
            return

        # Email configuration from environment
        smtp_server = os.environ.get("SMTP_SERVER", "localhost")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        recipients = os.environ.get("COVERAGE_ALERT_RECIPIENTS", "").split(",")

        if not recipients or not recipients[0]:
            print("No email recipients configured, skipping email alerts")
            return

        # Create email
        msg = MIMEMultipart()
        msg["From"] = smtp_user or "coverage@udl-framework.com"
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"Coverage Alert: {report.overall.coverage:.1f}%"

        # Email body
        body_lines = [
            "Coverage Alert Report",
            "=" * 50,
            "",
            f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Coverage: {report.overall.coverage:.1f}%",
            "",
            "Alerts:",
        ]

        for alert in alerts:
            body_lines.append(f"  {alert}")

        body_lines.extend(
            ["", "Full Report:", "-" * 20, self.generate_coverage_report(report)]
        )

        msg.attach(MIMEText("\n".join(body_lines), "plain"))

        # Send email
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            if smtp_user and smtp_password:
                server.starttls()
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
            server.quit()
            print(f"Alert email sent to {len(recipients)} recipients")
        except Exception as e:
            print(f"Failed to send email alert: {e}")

    def store_alerts(self, alerts: List[str], report: CoverageReport):
        """Store alerts in the database."""
        if not alerts:
            return

        conn = sqlite3.connect(self.db_path)

        for alert in alerts:
            alert_type = "CRITICAL" if "CRITICAL" in alert else "WARNING"
            conn.execute(
                """
                INSERT INTO coverage_alerts 
                (timestamp, alert_type, message, coverage_after)
                VALUES (?, ?, ?, ?)
            """,
                (
                    report.timestamp.isoformat(),
                    alert_type,
                    alert,
                    report.overall.coverage,
                ),
            )

        conn.commit()
        conn.close()
        print(f"Stored {len(alerts)} alerts")

    def run_monitoring_cycle(self, send_email: bool = False) -> bool:
        """Run a complete monitoring cycle."""
        try:
            # Run tests with coverage
            tests_passed, test_results = self.run_tests_with_coverage()

            # Collect coverage data
            report = self.collect_coverage_data()
            report.test_results = test_results

            # Store data
            self.store_coverage_data(report)

            # Check thresholds and generate alerts
            alerts = self.check_coverage_thresholds(report)

            # Store alerts
            self.store_alerts(alerts, report)

            # Print summary
            print("\nCoverage Monitoring Summary:")
            print(f"  Overall Coverage: {report.overall.coverage:.1f}%")
            print(f"  Tests Passed: {test_results.get('passed', 0)}")
            print(f"  Tests Failed: {test_results.get('failed', 0)}")
            print(f"  Alerts Generated: {len(alerts)}")

            if alerts:
                print("\nAlerts:")
                for alert in alerts:
                    print(f"  {alert}")

                if send_email:
                    self.send_alert_email(alerts, report)

            # Generate and save report
            report_text = self.generate_coverage_report(report)
            report_file = Path("coverage_report.md")
            report_file.write_text(report_text)
            print(f"\nDetailed report saved to: {report_file}")

            return len(alerts) == 0

        except Exception as e:
            print(f"Error during monitoring cycle: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="UDL Rating Framework Coverage Monitor"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Coverage threshold for warnings (default: 90.0)",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=85.0,
        help="Critical coverage threshold (default: 85.0)",
    )
    parser.add_argument(
        "--db-path",
        default="coverage_history.db",
        help="Path to coverage history database",
    )
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send email alerts for coverage issues",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with non-zero code if alerts are generated",
    )

    args = parser.parse_args()

    # Create monitor
    monitor = CoverageMonitor(
        db_path=args.db_path,
        threshold=args.threshold,
        critical_threshold=args.critical_threshold,
    )

    # Run monitoring cycle
    success = monitor.run_monitoring_cycle(send_email=args.send_email)

    # Exit with appropriate code
    if args.fail_on_alert and not success:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
