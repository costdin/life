use super::creature::*;
pub use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent},
    execute, queue,
    style::Color,
    style::{style, PrintStyledContent, SetForegroundColor},
    terminal::{self, ClearType},
    Command, ExecutableCommand, QueueableCommand, Result,
};
use rand::prelude::*;
use rayon::prelude::*;
use std::io::{stdout, Write};

pub struct World<const CHROMOSOME_COUNT: usize> {
    grid: Vec<Option<usize>>,
    width: usize,
    height: usize,
    age: usize,

    pub height_f: f64,
    pub width_f: f64,
    pub age_f: f64,
}

impl<const CHROMOSOME_COUNT: usize> World<CHROMOSOME_COUNT> {
    pub fn display(&self, creatures: &[Creature<CHROMOSOME_COUNT>]) {
        let mut console = stdout();
        for x in 0..self.width {
            for y in 0..self.height {
                console.queue(cursor::MoveTo(x as u16, y as u16));
                console.queue(PrintStyledContent(style(" ")));
            }
        }

        for c in creatures.iter() {
            console.queue(cursor::MoveTo(c.position.x as u16, c.position.y as u16));
            console.queue(PrintStyledContent(style("*")));
        }

        console.flush();
    }

    pub fn new(creatures: &mut Vec<Creature<CHROMOSOME_COUNT>>, height: usize, width: usize) -> World<CHROMOSOME_COUNT> {
        let mut rng = rand::thread_rng();
        let size = width * height;

        let mut grid: Vec<Option<usize>> = (0..size).step_by(1).map(|_| None).collect();

        for (ix, creature) in creatures.iter_mut().enumerate() {
            let mut i = rng.next_u32() as usize % size;

            while !grid[i].is_none() {
                i = rng.next_u32() as usize % size;
            }

            creature.position = Position {
                x: (i % height) as i16,
                y: (i / height) as i16,
            };
            grid[i] = Some(ix);
        }

        World {
            grid,
            width,
            height,
            age: 0,
            age_f: 0.,
            width_f: width as f64,
            height_f: height as f64,
        }
    }

    pub fn step(&mut self, creatures: &mut Vec<Creature<CHROMOSOME_COUNT>>, activate_kill: bool) {
        let (kills, moves, resp) = creatures
            .par_iter_mut()
            //.iter_mut()
            .filter(|c| c.alive)
            .map(|c| c.act(self))
            .flatten()
            .fold(
                || (vec![], vec![], vec![]),
                //(vec![], vec![], vec![]),
                |(mut kills, mut moves, mut resp), item| match item {
                    ActionResult::Move(src, dst) => {
                        moves.push((src, dst));
                        (kills, moves, resp)
                    }
                    ActionResult::Kill(k) => {
                        kills.push(k);
                        (kills, moves, resp)
                    }
                    ActionResult::SetResponsiveness(pos, r) => {
                        resp.push((pos, r));
                        (kills, moves, resp)
                    }
                },
            )
            .reduce(
                || (vec![], vec![], vec![]),
                |(mut kills, mut moves, mut resp), (k, m, r)| {
                    kills.extend_from_slice(&k);
                    moves.extend_from_slice(&m);
                    resp.extend_from_slice(&r);

                    (kills, moves, resp)
                },
            );

        if activate_kill {
            for kill in kills {
                let i = self.position_to_grid_index(&kill);
                if let Some(ix) = self.grid[i] {
                    creatures[ix].alive = false;
                    self.grid[i] = None;
                }
            }
        }

        /*
        for (pos, r) in resp {
            let i = self.position_to_grid_index(&pos);
            if let Some(ix) = self.grid[i] {
                if self.creatures[ix].alive {
                    self.creatures[ix].responsiveness = r;
                }
            }
        }*/

        for (source, destination) in moves {
            let dst = self.position_to_grid_index(&destination);
            if self.grid[dst].is_none() {
                let src = self.position_to_grid_index(&source);

                if let Some(ix) = self.grid[src] {
                    if creatures[ix].alive {
                        self.grid[src] = None;
                        self.grid[dst] = Some(ix);
                        creatures[ix].set_position(destination);
                    }
                }
            }
        }

        self.age += 1;
        self.age_f = self.age as f64;
    }

    #[inline]
    fn position_to_grid_index(&self, position: &Position) -> usize {
        self.get_grid_index(position.x as usize, position.y as usize)
    }

    #[inline]
    fn get_grid_index(&self, x: usize, y: usize) -> usize {
        x + y * self.width
    }

    #[inline]
    pub fn is_empty(&self, position: &Position) -> bool {
        self.grid[self.position_to_grid_index(position)].is_none()
    }

    #[inline]
    pub fn is_position_in_bounds(&self, position: &Position) -> bool {
        position.x >= 0
            && position.y >= 0
            && self.is_in_bounds(position.x as usize, position.y as usize)
    }

    #[inline]
    pub fn is_in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    pub fn get_neighborhood(&self, position: &Position, radius: usize) -> usize {
        let minx = if position.x as usize > radius {
            position.x as usize - radius
        } else {
            0
        };
        let maxx = usize::min(self.width - 1, position.x as usize + radius);

        let miny = if position.y as usize > radius {
            position.y as usize - radius
        } else {
            0
        };
        let maxy = usize::min(self.height - 1, position.y as usize + radius);

        let mut count = 0;
        for x in minx..maxx {
            for y in miny..maxy {
                if self.grid[self.get_grid_index(x, y)].is_some() {
                    count += 1;
                }
            }
        }

        count

        /*
        (minx..maxx)
        .zip(miny..maxy)
            .filter(|(x, y)| self.grid[self.get_grid_index(*x, *y)].is_some())
            .count()
        */
    }

    pub fn distance_next_creature(
        &self,
        position: &Position,
        direction: &Direction,
        infinity: usize,
    ) -> usize {
        let mut cnt = 1;
        let mut next = position.clone();
        loop {
            next = next.add_direction(direction);

            if !self.is_position_in_bounds(&next) {
                return infinity;
            }

            if !self.is_empty(&next) {
                return cnt;
            }

            cnt += 1;
        }
    }
}
