"""
test_parsing.py

Tests for the annotation-parsing logic in 01_parse_annotations.py.

Covers:
  - parse_annotation_file : reading label .txt files into event dicts
  - parseTextFile          : splitting filename into device/identifier/position/rep
  - getWavFileName         : stripping _label suffix to recover wav filename

Run with:  pytest test_parsing.py -v
"""

import sys
import importlib.util
import pytest
from pathlib import Path

# Module name starts with a digit — load directly
_mod_path = Path(__file__).resolve().parent / 'preprocessing' / '01_parse_annotations.py'
_spec = importlib.util.spec_from_file_location('parse_annotations', _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_annotation_file = _mod.parse_annotation_file
parseTextFile         = _mod.parseTextFile
getWavFileName        = _mod.getWavFileName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_txt(path: Path, lines: list[str]) -> None:
    path.write_text('\n'.join(lines))


# ---------------------------------------------------------------------------
# parse_annotation_file
# ---------------------------------------------------------------------------

class TestParseAnnotationFile:

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / 'empty_label.txt'
        write_txt(f, [])
        assert parse_annotation_file(f) == []

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / 'blank_label.txt'
        write_txt(f, ['', 'I 00:00:00.000 00:00:01.200', ''])
        events = parse_annotation_file(f)
        assert len(events) == 1

    def test_inhalation_parsed_correctly(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, ['I 00:00:00.000 00:00:01.200'])
        events = parse_annotation_file(f)
        assert len(events) == 1
        e = events[0]
        assert e['label']      == 'I'
        assert e['event_start'] == pytest.approx(0.0)
        assert e['event_end']   == pytest.approx(1.2)
        assert e['duration']    == pytest.approx(1.2)
        assert e['DAS'] == 0
        assert e['CAS'] == 0

    def test_exhalation_parsed_correctly(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, ['E 00:00:01.200 00:00:02.500'])
        events = parse_annotation_file(f)
        assert events[0]['label'] == 'E'
        assert events[0]['DAS']   == 0
        assert events[0]['CAS']   == 0

    def test_das_flag_set_for_D_event(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, ['D 00:00:00.500 00:00:00.700'])
        events = parse_annotation_file(f)
        assert events[0]['DAS'] == 1
        assert events[0]['CAS'] == 0
        assert events[0]['label'] == 'D'

    def test_cas_flag_set_for_wheeze(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, ['Wheeze 00:00:01.000 00:00:02.000'])
        events = parse_annotation_file(f)
        assert events[0]['CAS'] == 1
        assert events[0]['DAS'] == 0

    def test_cas_flag_set_for_rhonchi(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, ['Rhonchi 00:00:00.000 00:00:01.500'])
        events = parse_annotation_file(f)
        assert events[0]['CAS'] == 1
        assert events[0]['DAS'] == 0

    def test_cas_flag_set_for_stridor(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, ['Stridor 00:00:00.000 00:00:01.000'])
        events = parse_annotation_file(f)
        assert events[0]['CAS'] == 1
        assert events[0]['DAS'] == 0

    def test_multiple_events_parsed_in_order(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, [
            'I 00:00:00.000 00:00:01.200',
            'E 00:00:01.200 00:00:02.500',
            'D 00:00:01.500 00:00:01.600',
        ])
        events = parse_annotation_file(f)
        assert len(events) == 3
        assert [e['label'] for e in events] == ['I', 'E', 'D']

    def test_line_index_matches_file_line(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, [
            'I 00:00:00.000 00:00:01.000',   # line 0
            '',                               # line 1 — blank, skipped
            'E 00:00:01.000 00:00:02.000',   # line 2
        ])
        events = parse_annotation_file(f)
        assert events[0]['line_index'] == 0
        assert events[1]['line_index'] == 2

    def test_duration_calculated_correctly(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, ['I 00:00:01.500 00:00:03.000'])
        events = parse_annotation_file(f)
        assert events[0]['duration'] == pytest.approx(1.5)

    def test_malformed_line_too_few_fields_skipped(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, [
            'I 00:00:00.000',                        # only 2 fields
            'E 00:00:01.000 00:00:02.000',
        ])
        events = parse_annotation_file(f)
        assert len(events) == 1
        assert events[0]['label'] == 'E'

    def test_malformed_line_too_many_fields_skipped(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, [
            'I 00:00:00.000 00:00:01.000 extra',
            'E 00:00:01.000 00:00:02.000',
        ])
        events = parse_annotation_file(f)
        assert len(events) == 1
        assert events[0]['label'] == 'E'

    def test_invalid_time_format_skipped(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, [
            'I notaTime 00:00:01.000',
            'E 00:00:01.000 00:00:02.000',
        ])
        events = parse_annotation_file(f)
        assert len(events) == 1
        assert events[0]['label'] == 'E'

    def test_zero_duration_skipped(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, [
            'I 00:00:01.000 00:00:01.000',   # start == end
            'E 00:00:01.000 00:00:02.000',
        ])
        events = parse_annotation_file(f)
        assert len(events) == 1
        assert events[0]['label'] == 'E'

    def test_negative_duration_skipped(self, tmp_path):
        f = tmp_path / 'test_label.txt'
        write_txt(f, [
            'I 00:00:02.000 00:00:01.000',   # end before start
            'E 00:00:01.000 00:00:02.000',
        ])
        events = parse_annotation_file(f)
        assert len(events) == 1
        assert events[0]['label'] == 'E'

    def test_event_start_time_correct_seconds(self, tmp_path):
        # 1 minute 30.5 seconds = 90.5 s
        f = tmp_path / 'test_label.txt'
        write_txt(f, ['I 00:01:30.500 00:01:32.000'])
        events = parse_annotation_file(f)
        assert events[0]['event_start'] == pytest.approx(90.5)
        assert events[0]['event_end']   == pytest.approx(92.0)


# ---------------------------------------------------------------------------
# parseTextFile
# ---------------------------------------------------------------------------

class TestParseTextFile:

    def test_steth_device_identified(self):
        device, _, _, _ = parseTextFile('steth_2019_06_03_09_33_45_label')
        assert device == 'steth'

    def test_steth_identifier_joined(self):
        _, identifier, _, _ = parseTextFile('steth_2019_06_03_09_33_45_label')
        assert identifier == '20190603093345'   # all segments between steth_ and _label joined

    def test_steth_position_and_rep_are_NA(self):
        _, _, position, rep = parseTextFile('steth_2019_06_03_09_33_45_label')
        assert position == 'NA'
        assert rep      == 'NA'

    def test_trunc_device_identified(self):
        device, _, _, _ = parseTextFile('trunc_2019-06-03-09-33-45-Tc_1_label')
        assert device == 'trunc'

    def test_trunc_identifier_is_datetime_only(self):
        _, identifier, _, _ = parseTextFile('trunc_2019-06-03-09-33-45-Tc_1_label')
        assert identifier == '20190603093345'   # first 6 dash-separated segments (YYYYMMDDHHmmSS)

    def test_trunc_position_extracted(self):
        _, _, position, _ = parseTextFile('trunc_2019-06-03-09-33-45-Tc_1_label')
        assert position == 'Tc'

    def test_trunc_rep_extracted(self):
        _, _, _, rep = parseTextFile('trunc_2019-06-03-09-33-45-Tc_1_label')
        assert rep == '1'

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError):
            parseTextFile('unknown_2019_06_03_label')


# ---------------------------------------------------------------------------
# getWavFileName
# ---------------------------------------------------------------------------

class TestGetWavFileName:

    def test_steth_label_to_wav(self):
        wav = getWavFileName('steth_2019_06_03_09_33_45_label')
        assert wav == 'steth_2019_06_03_09_33_45.wav'

    def test_trunc_label_to_wav(self):
        wav = getWavFileName('trunc_2019-06-03-09-33-45-Tc_1_label')
        assert wav == 'trunc_2019-06-03-09-33-45-Tc_1.wav'

    def test_single_segment_raises(self):
        with pytest.raises(ValueError):
            getWavFileName('justoneword')
